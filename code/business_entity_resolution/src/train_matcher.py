"""Train the pairwise matcher on the dev candidates and score it on the validation slice.

  python train_matcher.py          # string features only
  python train_matcher.py --rev    # plus reverse features (needs full_train blocking)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
import decide
from features import FEATURES, build
from make_candidates import cand_path
from metric import load_ground_truth, score_report
from reverse import REV_FEATURES, add_reverse_features
from split import _bucket

FEATURE_VERSION = 2  # bump whenever features.py changes
FEATURE_CACHE = config.WORK_DIR / f"features_dev_v{FEATURE_VERSION}.parquet"
MODEL_PATH = config.WORK_DIR / "lgbm.txt"
DECISION_PATH = config.WORK_DIR / "decision.json"

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


def evaluate_rules(pred: pl.DataFrame, truth, ids) -> dict:
    """Score every decision rule on validation; return the best one's settings."""
    results = []

    def run(name, rows, **settings):
        rep = score_report(decide.to_sets(rows, ids), truth, ids)
        results.append((rep["f05"], name, settings))
        return rep

    print("\n-- single threshold --")
    for t in np.arange(0.4, 0.96, 0.05):
        rep = run("threshold", decide.by_threshold(pred, t), t=round(float(t), 2))
        print(f"t={t:.2f}  {fmt(rep)}")

    print("\n-- two thresholds (first pick / rest) --")
    grid = []
    for t1 in np.arange(0.5, 0.96, 0.05):
        for t2 in np.arange(0.3, 0.91, 0.05):
            if t2 <= t1:
                rep = run("two", decide.two_thresholds(pred, t1, t2),
                          t_first=round(float(t1), 2), t_rest=round(float(t2), 2))
                grid.append((rep["f05"], t1, t2))
    for f, t1, t2 in sorted(grid, reverse=True)[:5]:
        print(f"t_first={t1:.2f} t_rest={t2:.2f}  f05={f:.4f}")

    print("\n-- expected-F set selection --")
    print("expected_f            ", fmt(run("expected_f", decide.expected_f(pred))))
    print("one_owner + expected_f", fmt(run("owner_expected_f",
                                            decide.expected_f(decide.one_owner(pred)))))

    best = max(results, key=lambda r: r[0])
    print(f"\nBEST: {best[1]} {best[2]}  f05={best[0]:.4f}")
    return {"rule": best[1], **best[2], "val_f05": best[0]}


def main(use_rev: bool):
    df = load_features()
    feats = list(FEATURES)
    if use_rev:
        t0 = time.time()
        df = add_reverse_features(df, "train")
        feats += REV_FEATURES
        print(f"reverse features added in {time.time() - t0:.0f}s")
    train = df.filter(pl.col("part") == "train")
    val = df.filter(pl.col("part") == "val")
    # early-stopping slice carved from TRAINING entities (validation stays untouched)
    stop = train["s1_id"].map_elements(lambda e: _bucket(e) >= 0.16, return_dtype=pl.Boolean)
    fit, es = train.filter(~stop), train.filter(stop)
    print(f"fit rows={fit.height:,}  early-stop rows={es.height:,}  val rows={val.height:,}  "
          f"pos rate={train['label'].mean():.3f}  features={len(feats)}")

    t0 = time.time()
    model = lgb.train(
        PARAMS,
        lgb.Dataset(fit.select(feats).to_numpy(), fit["label"].to_numpy(), feature_name=feats),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(es.select(feats).to_numpy(), es["label"].to_numpy())],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    model.save_model(str(MODEL_PATH), num_iteration=model.best_iteration)
    print(f"trained {model.best_iteration} rounds in {time.time() - t0:.0f}s")

    pred = val.select("s1_id", "cand_id").with_columns(
        pl.Series("p", model.predict(val.select(feats).to_numpy(),
                                     num_iteration=model.best_iteration)))
    truth = load_ground_truth(config.ground_truth_parquet())
    # validation entities with zero candidates still count (as empty predictions)
    ids = list(set(pl.read_parquet(cand_path("dev"), columns=["s1_id", "part"])
                     .filter(pl.col("part") == "val")["s1_id"]))
    decision = evaluate_rules(pred, truth, ids)
    DECISION_PATH.write_text(json.dumps(decision, indent=1))

    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])
    total = sum(g for _, g in imp)
    print("\ntop features by gain:", ", ".join(f"{f} {g / total:.1%}" for f, g in imp[:15]))


if __name__ == "__main__":
    main("--rev" in sys.argv)
