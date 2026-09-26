"""Turn candidate probabilities into a match set per S1 entity."""
from collections import defaultdict

import polars as pl

BETA2 = 0.25


def to_sets(rows: pl.DataFrame, ids) -> dict:
    """rows(s1_id, cand_id) -> {s1_id: set}, with an empty set for every id in `ids`."""
    out = {i: set() for i in ids}
    for s, c in zip(rows["s1_id"], rows["cand_id"]):
        out[s].add(c)
    return out


def by_threshold(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    return pred.filter(pl.col("p") >= t).select("s1_id", "cand_id")


def one_owner(pred: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 record belongs to at most one S1 entity (holds in 100% of ground truth):
    keep only the pair with the highest probability for every candidate record."""
    return pred.filter(pl.col("p") == pl.col("p").max().over("cand_id"))


def expected_f(pred: pl.DataFrame, min_p: float = 0.0) -> pl.DataFrame:
    """Per S1, keep the probability-sorted prefix that maximizes plug-in expected F0.5:
    F = 1.25*tp / (1.25*tp + 0.25*fn + fp) with tp = sum of kept p, fn = sum of dropped p,
    fp = kept - tp. The empty set is chosen when P(no match) = prod(1-p) beats it."""
    df = (pred.filter(pl.col("p") >= min_p)
              .sort(["s1_id", "p"], descending=[False, True])
              .with_columns(
                  pl.col("p").cum_sum().over("s1_id").alias("tp"),
                  pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
                  pl.col("p").sum().over("s1_id").alias("T")))
    df = df.with_columns(
        (1.25 * pl.col("tp") / (1.25 * pl.col("tp") + BETA2 * (pl.col("T") - pl.col("tp"))
                                + (pl.col("k") - pl.col("tp")))).alias("F"))
    # P(no match) over ALL candidates of the entity (unfiltered)
    p_empty = pred.group_by("s1_id").agg((1 - pl.col("p")).product().alias("F_empty"))
    best = (df.group_by("s1_id").agg(pl.col("F").max().alias("F_best"),
                                     pl.col("k").get(pl.col("F").arg_max()).alias("k_best"))
              .join(p_empty, on="s1_id"))
    keep = best.filter(pl.col("F_best") > pl.col("F_empty")).select("s1_id", "k_best")
    return (df.join(keep, on="s1_id").filter(pl.col("k") <= pl.col("k_best"))
              .select("s1_id", "cand_id"))
