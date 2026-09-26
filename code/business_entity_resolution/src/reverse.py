"""Reverse (candidate-side) features from full-density blocking.

Every S2/S3 record belongs to at most one S1 entity. With every S1 entity queried, we
can ask of each candidate record: how many S1 entities retrieved it, and is this S1 its
best match? That separates sibling decoys (a different S1 claims them more strongly)
from true copies, and unique no-address records from namesake ties.

Only (record, score) is aggregated, with integer keys, so 88M rows fit in memory.
"""
import polars as pl

import config

REV_FEATURES = ["rev_n", "rev_gap", "rev_margin", "rev_n_close", "rev_is_best"]
CLOSE = 0.02  # scores within this of the best count as a near-tie


def full_dir(split: str):
    return config.WORK_DIR / f"full_{split}"


def cand_key(source: pl.Expr, cand_id: pl.Expr) -> pl.Expr:
    """Integer key unique across S2/S3 (ids are 'S2-<digits>' / 'S3-<digits>')."""
    return source.cast(pl.Int64) * 10_000_000_000 + cand_id.str.slice(3).cast(pl.Int64)


def reverse_stats(split: str) -> pl.DataFrame:
    """Per candidate record: rev_n, rev_best, rev_second, rev_n_close (cached)."""
    path = config.WORK_DIR / f"reverse_{split}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    lf = (pl.scan_parquet(str(full_dir(split) / "*.parquet"))
            .select(cand_key(pl.col("source"), pl.col("cand_id")).alias("ck"), "score"))
    stats = (lf.group_by("ck")
               .agg(pl.len().cast(pl.Int32).alias("rev_n"),
                    pl.col("score").max().alias("rev_best"),
                    pl.col("score").top_k(2).min().alias("rev_second"))
               .collect(engine="streaming"))
    close = (lf.join(stats.lazy().select("ck", "rev_best"), on="ck")
               .filter(pl.col("score") >= pl.col("rev_best") - CLOSE)
               .group_by("ck").agg(pl.len().cast(pl.Int32).alias("rev_n_close"))
               .collect(engine="streaming"))
    stats = stats.join(close, on="ck", how="left")
    stats.write_parquet(path)
    return stats


def add_reverse_features(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """pairs needs source, cand_id, score. Adds REV_FEATURES."""
    stats = reverse_stats(split)
    out = (pairs.with_columns(cand_key(pl.col("source"), pl.col("cand_id")).alias("ck"))
                .join(stats, on="ck", how="left"))
    is_best = pl.col("score") >= pl.col("rev_best") - 1e-6
    return out.with_columns(
        is_best.cast(pl.Int8).alias("rev_is_best"),
        (pl.col("rev_best") - pl.col("score")).alias("rev_gap"),
        pl.when(is_best).then(pl.col("score") - pl.col("rev_second"))
          .otherwise(pl.col("score") - pl.col("rev_best")).alias("rev_margin"),
    ).drop("ck", "rev_best", "rev_second")
