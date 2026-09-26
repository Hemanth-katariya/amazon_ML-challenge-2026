"""Score the test candidates and write the two submission files.

Two-stage (stack.py) when lgbm_stage2.txt exists, else single model (train_matcher.py).
Inputs : WORK_DIR/full_test/*.parquet (full_blocking.py test), model(s), decision json
Outputs: OUTPUT_DIR/matching_results.tsv, OUTPUT_DIR/candidate_pairs.tsv
"""
import json
import time

import lightgbm as lgb
import polars as pl

import config
import decide
from features import Tables, attach_text, iter_chunks, pair_features
from reverse import add_reverse_features, full_dir
from stack import STAGE1_PATH, STAGE2_PATH, stage2_features

PRED_PATH = config.WORK_DIR / "test_pred.parquet"


def id_lists(rows: pl.DataFrame, s1_ids: pl.Series, col: str) -> pl.DataFrame:
    """One row per test S1 entity with a comma-joined (possibly empty) id list."""
    lists = (rows.unique(["s1_id", "cand_id"])
                 .group_by("s1_id").agg(pl.col("cand_id").sort().str.join(",").alias(col)))
    return (pl.DataFrame({"source1_entity_id": s1_ids})
              .join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
              .with_columns(pl.col(col).fill_null("")))


def write_tsv(df: pl.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never")


def _predict(model: lgb.Booster, df: pl.DataFrame):
    return model.predict(df.select(pl.col(model.feature_name()).cast(pl.Float32)).to_numpy())


class Scorer:
    def __init__(self):
        if STAGE2_PATH.exists():
            self.m1 = lgb.Booster(model_file=str(STAGE1_PATH))
            self.m2 = lgb.Booster(model_file=str(STAGE2_PATH))
            self.decision = json.loads((config.WORK_DIR / "decision_stage2.json").read_text())
        else:
            self.m1 = lgb.Booster(model_file=str(config.WORK_DIR / "lgbm.txt"))
            self.m2 = None
            self.decision = json.loads((config.WORK_DIR / "decision.json").read_text())
        self.use_rev = any(f.startswith("rev_") for f in self.m1.feature_name())

    def __call__(self, f: pl.DataFrame) -> pl.DataFrame:
        """f = pair features for whole S1 groups -> frame(s1_id, cand_id, p)."""
        if self.use_rev:
            f = add_reverse_features(f, "test")
        p1 = _predict(self.m1, f)
        if self.m2 is None:
            return f.select("s1_id", "cand_id").with_columns(pl.Series("p", p1))
        f = f.with_columns(
            pl.Series("p1", p1),
            pl.col("core_n_b").alias("core_key"),
            pl.col("addr_n_b").str.extract_all(r"\d+").list.unique().list.sort().list.join(" ")
              .alias("nums_key"))
        f = stage2_features(f)
        return f.select("s1_id", "cand_id").with_columns(pl.Series("p", _predict(self.m2, f)))


def main(cand_dir=None, out_dir=None):
    t0 = time.time()
    cand_dir = cand_dir or full_dir("test")
    out_dir = out_dir or config.OUTPUT_DIR
    cands = pl.read_parquet(str(cand_dir / "*.parquet"))
    scorer = Scorer()
    print(f"candidates: {cands.height:,} rows, {cands['s1_id'].n_unique():,} S1; "
          f"two-stage={scorer.m2 is not None} reverse={scorer.use_rev} "
          f"decision={scorer.decision}", flush=True)

    tables = Tables("test")
    preds = []
    for i, chunk in enumerate(iter_chunks(cands, 100_000)):
        preds.append(scorer(pair_features(attach_text(chunk, tables))))
        print(f"  chunk {i}: {chunk.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
    pred = pl.concat(preds)
    pred.write_parquet(PRED_PATH)

    matches = decide.apply(pred, scorer.decision)
    s1_ids = pl.read_parquet(config.parquet_path("test", 1), columns=["entity_id"])["entity_id"]
    write_tsv(id_lists(matches, s1_ids, "matched_entity_ids"), out_dir / "matching_results.tsv")
    write_tsv(id_lists(cands, s1_ids, "candidate_entity_ids"), out_dir / "candidate_pairs.tsv")
    n_match = matches["s1_id"].n_unique()
    print(f"{matches.height:,} matches for {n_match:,}/{len(s1_ids):,} S1 entities; "
          f"wrote {out_dir} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    args = [Path(a) for a in sys.argv[1:3]]
    main(*args)
