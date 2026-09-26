"""Generate blocking candidates and save them, labelled when ground truth exists.

  python make_candidates.py dev   -> train sample + validation slice (labelled)
  python make_candidates.py extra -> a second, disjoint train sample (needs full_train)
  python make_candidates.py test  -> every test S1 entity
"""
import sys
import time

import polars as pl

import config
from blocking import generate
from split import VAL_FRACTION, bucket_range, s1_frame

TRAIN_SAMPLE = (VAL_FRACTION, VAL_FRACTION + 0.07)  # ~154k training entities
EXTRA_SAMPLE = (VAL_FRACTION + 0.07, VAL_FRACTION + 0.14)  # another ~155k, disjoint
VAL_SLICE = (0.0, 0.025)                             # ~55k of the validation entities


def cand_path(name: str):
    return config.WORK_DIR / f"cands_{name}.parquet"


def label(cands: pl.DataFrame) -> pl.DataFrame:
    gt = (pl.read_parquet(config.ground_truth_parquet())
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True).drop_nulls()
            .select(pl.col("source1_entity_id").alias("s1_id"),
                    pl.col("matched_entity_ids").alias("cand_id"),
                    pl.lit(1, pl.Int8).alias("label")))
    return (cands.join(gt, on=["s1_id", "cand_id"], how="left")
                 .with_columns(pl.col("label").fill_null(0)))


def main(which: str):
    t0 = time.time()
    if which == "dev":
        s1 = s1_frame("train")
        train_ids = bucket_range(s1, *TRAIN_SAMPLE)
        val_ids = bucket_range(s1, *VAL_SLICE)
        print(f"train sample={len(train_ids):,}  val slice={len(val_ids):,}")
        cands = generate("train", train_ids | val_ids)
        part = pl.when(pl.col("s1_id").is_in(list(val_ids))).then(pl.lit("val")).otherwise(pl.lit("train"))
        cands = label(cands).with_columns(part.alias("part"))
    elif which == "extra":
        # more training entities, taken from the full-density blocking output
        from reverse import full_dir
        ids = bucket_range(s1_frame("train"), *EXTRA_SAMPLE)
        print(f"extra training sample={len(ids):,}")
        cands = (pl.scan_parquet(str(full_dir("train") / "*.parquet"))
                   .filter(pl.col("s1_id").is_in(list(ids))).collect())
        cands = label(cands).with_columns(pl.lit("train").alias("part"))
    elif which == "test":
        cands = generate("test")
    else:
        raise SystemExit(f"unknown target {which!r}; use dev or test")
    cands.write_parquet(cand_path(which))
    print(f"saved {cands.height:,} candidate rows -> {cand_path(which)} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
