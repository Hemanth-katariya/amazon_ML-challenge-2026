"""Measure blocking quality on the validation split.

Reports, per key and for the union:
  * pair recall      : share of true (S1, match) pairs present in the candidates
  * ceiling F0.5     : score if the matcher were perfect within the candidates
  * candidates / S1  : size of the set the matcher must score
"""
import sys
import time

import polars as pl

import config
from blocking import DEFAULT_KEYS, generate
from metric import load_ground_truth, score_report
from split import random_split, s1_frame


def evaluate(cands: pl.DataFrame, truth, ids, label: str):
    grouped = cands.group_by("s1_id").agg(pl.col("cand_id").unique())
    cand = {s: set(c) for s, c in zip(grouped["s1_id"], grouped["cand_id"])}
    tp = sum(len(truth[i] & cand.get(i, set())) for i in ids)
    n_true = sum(len(truth[i]) for i in ids)
    ceiling = {i: truth[i] & cand.get(i, set()) for i in ids}
    rep = score_report(ceiling, truth, ids)
    n_cand = sum(len(cand.get(i, ())) for i in ids) / len(ids)
    print(f"  {label:<22} pair_recall={tp / n_true:.4f}  ceiling_f05={rep['f05']:.4f}  "
          f"cands/S1={n_cand:.1f}")
    return tp / n_true


def main(frac=None):
    truth = load_ground_truth(config.ground_truth_parquet())
    _, val = random_split(s1_frame())
    if frac:  # quick run on a slice of the validation entities
        val = set(sorted(val)[: int(len(val) * frac)])
    print(f"validation S1 entities: {len(val):,}")
    t0 = time.time()
    cands = generate("train", val)
    print(f"blocking time: {time.time() - t0:.0f}s, candidate rows: {cands.height:,}")
    cands.write_parquet(config.WORK_DIR / "val_candidates.parquet")

    ids = [i for i in val if i in truth]
    for key in DEFAULT_KEYS:
        for k in sorted({5, 10, key.k}):
            evaluate(cands.filter((pl.col("key") == key.name) & (pl.col("rank") < k)),
                     truth, ids, f"{key.name} k={k}")
    if len(DEFAULT_KEYS) > 1:
        for k in (5, 10, 20):
            evaluate(cands.filter(pl.col("rank") < k), truth, ids, f"UNION k={k}")
    # recall by country
    s1 = s1_frame().select("entity_id", "country")
    for country in ("US", "India"):
        cids = set(s1.filter(pl.col("country") == country)["entity_id"]) & set(ids)
        evaluate(cands, truth, list(cids), f"ALL {country}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
