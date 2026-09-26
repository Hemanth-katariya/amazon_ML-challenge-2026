"""Deterministic train/validation split of Source-1 entities.

The split is by Source-1 entity: an entity and all its S2/S3 matches land on the same
side, so validation always runs the full pipeline (blocking -> matching) on entities
the model never saw.

Two schemes:
  * random    : hold out VAL_FRACTION of S1 entities in every country.
  * country   : hold out one whole country (India as a stand-in for unseen France).
"""
import hashlib

import polars as pl

import config

VAL_FRACTION = 0.10


def _bucket(entity_id: str) -> float:
    # Stable across runs and machines (unlike Python's hash()).
    h = hashlib.md5(entity_id.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def s1_frame(split: str = "train") -> pl.DataFrame:
    return pl.read_parquet(config.parquet_path(split, 1))


def random_split(s1: pl.DataFrame, frac: float = VAL_FRACTION):
    """Return (train_ids, val_ids) as sets of S1 entity ids."""
    val = {e for e in s1["entity_id"] if _bucket(e) < frac}
    return set(s1["entity_id"]) - val, val


def bucket_range(s1: pl.DataFrame, lo: float, hi: float):
    """S1 ids whose stable bucket falls in [lo, hi). Buckets below VAL_FRACTION are
    validation, so e.g. [0.10, 0.17) is a ~7% training sample disjoint from it."""
    return {e for e in s1["entity_id"] if lo <= _bucket(e) < hi}


def country_split(s1: pl.DataFrame, holdout: str):
    """Train on every other country, validate on `holdout` (e.g. 'India')."""
    val = set(s1.filter(pl.col("country") == holdout)["entity_id"])
    return set(s1["entity_id"]) - val, val


if __name__ == "__main__":
    s1 = s1_frame()
    tr, va = random_split(s1)
    print(f"random split : train={len(tr):,} val={len(va):,} ({len(va)/s1.height:.1%})")
    by_country = s1.filter(pl.col("entity_id").is_in(list(va))).group_by("country").len()
    print(by_country.sort("country"))
    tr, va = country_split(s1, "India")
    print(f"country split: train(US)={len(tr):,} val(India)={len(va):,}")
