"""Train the pairwise matcher on the dev candidates and score it on the validation slice."""
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import decide
from features import FEATURES, build
from make_candidates import cand_path
from metric import load_ground_truth, score_report
from split import _bucket

FEATURE_VERSION = 2  # bump whenever features.py changes
FEATURE_CACHE = config.WORK_DIR / f"features_dev_v{FEATURE_VERSION}.parquet"
MODEL_PATH = config.WORK_DIR / "lgbm.txt"

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=-1, verbose=-1)


def load_features() -> pl.DataFrame:
    if FEATURE_CACHE.exists():
        return pl.read_parquet(FEATURE_CACHE)
    t0 = time.time()
    df = build(pl.read_parquet(cand_path("dev")), "train")
    df.select("s1_id", "cand_id", "part", "label", *FEATURES).write_parquet(FEATURE_CACHE)
    print(f"features: {df.height:,} rows in {time.time() - t0:.0f}s")
    return pl.read_parquet(FEATURE_CACHE)


def fmt(rep):
    return "  ".join(f"{k}={v:.4f}" for k, v in rep.items() if k != "n")


def main():
    df = load_features()
    train = df.filter(pl.col("part") == "train")
    val = df.filter(pl.col("part") == "val")
    # early-stopping slice carved from TRAINING entities (validation stays untouched)
    stop = train["s1_id"].map_elements(lambda e: _bucket(e) >= 0.16, return_dtype=pl.Boolean)
    fit, es = train.filter(~stop), train.filter(stop)
    print(f"fit rows={fit.height:,}  early-stop rows={es.height:,}  val rows={val.height:,}  "
          f"pos rate={train['label'].mean():.3f}")

    t0 = time.time()
    model = lgb.train(
        PARAMS,
        lgb.Dataset(fit.select(FEATURES).to_numpy(), fit["label"].to_numpy(), feature_name=FEATURES),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(es.select(FEATURES).to_numpy(), es["label"].to_numpy())],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    model.save_model(str(MODEL_PATH))
    print(f"trained {model.best_iteration} rounds in {time.time() - t0:.0f}s")

    pred = val.select("s1_id", "cand_id").with_columns(
        pl.Series("p", model.predict(val.select(FEATURES).to_numpy(),
                                     num_iteration=model.best_iteration)))
    truth = load_ground_truth(config.ground_truth_parquet())
    ids = pred["s1_id"].unique().to_list()
    # validation entities with zero candidates still count (as empty predictions)
    val_ids = set(pl.read_parquet(cand_path("dev"), columns=["s1_id", "part"])
                    .filter(pl.col("part") == "val")["s1_id"])
    print(f"val entities with candidates: {len(ids):,}")
    ids = list(val_ids)

    print("\n-- threshold sweep --")
    best_t, best = None, -1
    for t in np.arange(0.2, 0.95, 0.05):
        rep = score_report(decide.to_sets(decide.by_threshold(pred, t), ids), truth, ids)
        if rep["f05"] > best:
            best_t, best = t, rep["f05"]
        print(f"t={t:.2f}  {fmt(rep)}")
    print(f"best threshold {best_t:.2f}: f05={best:.4f}")

    print("\n-- expected-F set selection --")
    rep = score_report(decide.to_sets(decide.expected_f(pred), ids), truth, ids)
    print("expected_f            ", fmt(rep))
    rep = score_report(decide.to_sets(decide.expected_f(decide.one_owner(pred)), ids), truth, ids)
    print("one_owner + expected_f", fmt(rep))

    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])
    total = sum(g for _, g in imp)
    print("\ntop features by gain:", ", ".join(f"{f} {g / total:.1%}" for f, g in imp[:12]))


if __name__ == "__main__":
    main()
