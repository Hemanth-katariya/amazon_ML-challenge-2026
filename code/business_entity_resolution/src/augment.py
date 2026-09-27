"""Decoy-branch augmentation: make training/validation candidate lists look like test.

The test set has ~2x the unmatched (decoy) records per S1 entity of train (2.3 vs 1.2)
and they come as *branches*: several records sharing a near-miss address (another house
number, an extra name word) that back each other up. Group-support features learned on
train (where a decoy usually stands alone) then vote FOR a decoy branch.

We simulate test by cloning each pure decoy candidate (label 0 and owned by no S1 entity)
into the other source with identical pair features, then recomputing the group features.
"""
import numpy as np
import polars as pl

GROUP = ["sup_num", "sup_core", "num_is_mode", "core_is_mode"]


def matched_records(gt_path) -> pl.Series:
    return (pl.read_parquet(gt_path).select(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids").drop_nulls()
              .filter(pl.col("matched_entity_ids") != "")["matched_entity_ids"])


def clone_decoys(df: pl.DataFrame, matched: pl.Series, rate: float = 1.0, seed: int = 0) -> pl.DataFrame:
    """df: whole S1 groups with label, source, nums_key, core_key and pair features."""
    decoy = (pl.col("label") == 0) & ~pl.col("cand_id").is_in(matched.implode())
    pool = df.filter(decoy)
    if rate < 1.0:
        pool = pool.filter(pl.Series(np.random.default_rng(seed).random(pool.height) < rate))
    clones = pool.with_columns((pl.col("cand_id") + "c").alias("cand_id"),
                               (5 - pl.col("source")).cast(df["source"].dtype).alias("source"))
    return recompute_group(pl.concat([df, clones], how="vertical_relaxed"))


BRANCH = ["num_agree", "ent_n_agree", "grp_n_agree", "conflict_grp_size", "ent_n_branches"]


def branch_features(df: pl.DataFrame) -> pl.DataFrame:
    """Competing-branch signals. A decoy branch shares its own house numbers, which
    conflict with S1's; the real copies usually agree with S1. Needs nums_key,
    num_conflict (whole S1 groups)."""
    agree = (pl.col("nums_key") != "") & (pl.col("num_conflict") == 0)
    conflict = (pl.col("nums_key") != "") & (pl.col("num_conflict") > 0)
    out = df.with_columns(agree.cast(pl.Int8).alias("num_agree"),
                          pl.len().over("s1_id", "nums_key").alias("_g"))
    out = out.with_columns(
        pl.col("num_agree").sum().over("s1_id").cast(pl.Float32).alias("ent_n_agree"),
        pl.col("num_agree").sum().over("s1_id", "nums_key").cast(pl.Float32).alias("grp_n_agree"),
        pl.when(conflict).then(pl.col("_g")).otherwise(0).cast(pl.Float32).alias("conflict_grp_size"),
    )
    branches = (out.filter(conflict & (pl.col("_g") >= 2)).group_by("s1_id")
                   .agg(pl.col("nums_key").n_unique().cast(pl.Float32).alias("ent_n_branches")))
    return (out.join(branches, on="s1_id", how="left", maintain_order="left")
               .with_columns(pl.col("ent_n_branches").fill_null(0.0),
                             pl.col("num_agree").cast(pl.Float32))
               .drop("_g"))


def recompute_group(df: pl.DataFrame) -> pl.DataFrame:
    """Same definitions as features.pair_features (nums_key == its _nums)."""
    has = pl.col("nums_key") != ""
    out = df.with_columns(pl.len().over("s1_id", "nums_key").alias("_n_nums"),
                          pl.len().over("s1_id", "core_key").alias("_n_core"))
    out = out.with_columns(
        pl.when(has).then(pl.col("_n_nums") - 1).otherwise(0).cast(df["sup_num"].dtype).alias("sup_num"),
        (pl.col("_n_core") - 1).cast(df["sup_core"].dtype).alias("sup_core"),
        (has & (pl.col("_n_nums") == pl.col("_n_nums").filter(has).max().over("s1_id")))
        .cast(df["num_is_mode"].dtype).alias("num_is_mode"),
        (pl.col("_n_core") == pl.col("_n_core").max().over("s1_id"))
        .cast(df["core_is_mode"].dtype).alias("core_is_mode"),
    )
    return out.drop("_n_nums", "_n_core")
