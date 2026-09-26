"""Two-stage matcher: stage 2 sees stage-1 probabilities of the entity's OTHER candidates.

Most remaining loss is "one wrong in the group": a missed copy next to confident ones,
or a lone decoy next to three agreeing true copies. Stage-2 features summarize stage-1
probabilities within each S1 entity's candidate list (support from candidates that share
the same house numbers / core name, best probability in the other source, ...).

Stage-1 probabilities on training entities are out-of-fold (a model that never saw the
entity), so stage 2 trains on the same kind of probabilities it will see at test time.

  python stack.py          # stage 1 on string features
  python stack.py --rev    # stage 1 also uses reverse features
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from features import FEATURES
from make_candidates import cand_path
from metric import load_ground_truth
from reverse import REV_FEATURES, add_reverse_features
from split import _bucket
from train_matcher import FEATURE_CACHE, PARAMS, evaluate_rules

K_FOLDS = 3
STAGE2_EXTRA = ["p1", "p1_rank", "p1_gap", "p1_n_above", "p1_sum",
                "sup_num_p", "sup_core_p", "p1_other_src_max", "p1_same_src_max_other"]
STAGE1_PATH = config.WORK_DIR / "lgbm_stage1.txt"
STAGE2_PATH = config.WORK_DIR / "lgbm_stage2.txt"


def stage2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Summaries of stage-1 probabilities within each S1 entity's candidate list.
    Needs columns: s1_id, source, p1, nums_key, core_key (whole S1 groups)."""
    df = df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").alias("p1_rank"),
        (pl.col("p1").max().over("s1_id") - pl.col("p1")).alias("p1_gap"),
        (pl.col("p1") > 0.5).sum().over("s1_id").alias("p1_n_above"),
        pl.col("p1").sum().over("s1_id").alias("p1_sum"),
        # support weighted by probability: other candidates sharing numbers / core name
        pl.when(pl.col("nums_key") != "")
          .then(pl.col("p1").sum().over("s1_id", "nums_key") - pl.col("p1"))
          .otherwise(0.0).alias("sup_num_p"),
        (pl.col("p1").sum().over("s1_id", "core_key") - pl.col("p1")).alias("sup_core_p"),
    )
    by_src = df.group_by("s1_id", "source").agg(pl.col("p1").max().alias("_src_max"))
    other = by_src.with_columns((5 - pl.col("source")).alias("source")).rename(
        {"_src_max": "p1_other_src_max"})  # source 2 <-> 3
    df = df.join(other, on=["s1_id", "source"], how="left").with_columns(
        pl.col("p1_other_src_max").fill_null(0.0))
    # best OTHER candidate in the same source
    top2 = df.group_by("s1_id", "source").agg(pl.col("p1").top_k(2).alias("_t"))
    df = df.join(top2, on=["s1_id", "source"], how="left").with_columns(
        pl.when(pl.col("p1") >= pl.col("_t").list.first())
          .then(pl.col("_t").list.get(1, null_on_oob=True))
          .otherwise(pl.col("_t").list.first()).fill_null(0.0).alias("p1_same_src_max_other"))
    return df.drop("_t")


def group_keys(split: str, df: pl.DataFrame) -> pl.DataFrame:
    """Attach nums_key (sorted house-number set of the candidate) and core_key."""
    from blocking import load_norm
    txt = pl.concat([load_norm(split, s).select("entity_id", "core_n", "addr_n") for s in (2, 3)])
    txt = txt.filter(pl.col("entity_id").is_in(df["cand_id"].unique().implode())).select(
        pl.col("entity_id").alias("cand_id"),
        pl.col("core_n").alias("core_key"),
        pl.col("addr_n").str.extract_all(r"\d+").list.unique().list.sort().list.join(" ")
          .alias("nums_key"))
    return df.join(txt, on="cand_id", how="left")


def fit(X, y, feats, rounds=None, es=None):
    kw = dict(num_boost_round=rounds or 3000)
    if es is not None:
        kw.update(valid_sets=[lgb.Dataset(es[0], es[1])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats), **kw)


def main(use_rev: bool):
    t0 = time.time()
    df = pl.read_parquet(FEATURE_CACHE)
    feats1 = list(FEATURES)
    if use_rev:
        df = add_reverse_features(df, "train")
        feats1 += REV_FEATURES
    df = group_keys("train", df)
    train = df.filter(pl.col("part") == "train")
    val = df.filter(pl.col("part") == "val")
    fold = np.array([int(_bucket(e) * 1000) % K_FOLDS for e in train["s1_id"]])
    stop = np.array([_bucket(e) >= 0.16 for e in train["s1_id"]])
    X, y = train.select(pl.col(feats1).cast(pl.Float32)).to_numpy(), train["label"].to_numpy()

    # stage 1 on all training entities -> used for validation (and test)
    m1 = fit(X[~stop], y[~stop], feats1, es=(X[stop], y[stop]))
    rounds = m1.best_iteration
    m1.save_model(str(STAGE1_PATH), num_iteration=rounds)
    print(f"stage 1: {rounds} rounds ({time.time() - t0:.0f}s)", flush=True)
    val = val.with_columns(pl.Series("p1", m1.predict(val.select(pl.col(feats1).cast(pl.Float32)).to_numpy())))

    # out-of-fold stage-1 probabilities for training entities
    oof = np.zeros(train.height)
    for k in range(K_FOLDS):
        mk = fit(X[fold != k], y[fold != k], feats1, rounds=rounds)
        oof[fold == k] = mk.predict(X[fold == k])
        print(f"  fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    train = train.with_columns(pl.Series("p1", oof))

    train, val = stage2_features(train), stage2_features(val)
    feats2 = feats1 + STAGE2_EXTRA
    X2 = train.select(pl.col(feats2).cast(pl.Float32)).to_numpy()
    m2 = fit(X2[~stop], y[~stop], feats2, es=(X2[stop], y[stop]))
    m2.save_model(str(STAGE2_PATH), num_iteration=m2.best_iteration)
    print(f"stage 2: {m2.best_iteration} rounds ({time.time() - t0:.0f}s)", flush=True)

    truth = load_ground_truth(config.ground_truth_parquet())
    ids = list(set(pl.read_parquet(cand_path("dev"), columns=["s1_id", "part"])
                     .filter(pl.col("part") == "val")["s1_id"]))
    print("\n===== stage 1 alone =====")
    evaluate_rules(val.select("s1_id", "cand_id", pl.col("p1").alias("p")), truth, ids)
    print("\n===== stage 2 =====")
    pred = val.select("s1_id", "cand_id").with_columns(
        pl.Series("p", m2.predict(val.select(pl.col(feats2).cast(pl.Float32)).to_numpy())))
    pred.write_parquet(config.WORK_DIR / "val_pred_stage2.parquet")
    decision = evaluate_rules(pred, truth, ids)
    decision.update(stacked=True, rev=use_rev)
    (config.WORK_DIR / "decision_stage2.json").write_text(json.dumps(decision, indent=1))
    imp = sorted(zip(feats2, m2.feature_importance("gain")), key=lambda x: -x[1])
    total = sum(g for _, g in imp)
    print("\nstage-2 top features:", ", ".join(f"{f} {g / total:.1%}" for f, g in imp[:12]))


if __name__ == "__main__":
    main("--rev" in sys.argv)
