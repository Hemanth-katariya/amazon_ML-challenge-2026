"""Compute pair features for the test candidates and cache them to disk (restartable).

Blocking parts are named s<source>_<country>_<key>_<start>.parquet; the S2 and S3 parts
with the same country and start offset hold the candidates of the same S1 entities, so
they form one group of whole S1 candidate lists. A group is processed once both parts
exist (and have not been touched for a minute), so this can run while blocking is still
producing parts. Features do not depend on the model: rescoring with a new model only
needs predict_test.py.

  python test_features.py            # process every finished group, then exit
"""
import time
from collections import defaultdict

import polars as pl

import config
from features import FEATURES, Tables, attach_text, iter_chunks, pair_features
from reverse import full_dir

FEATS_DIR = config.WORK_DIR / "test_feats"
KEEP = ["s1_id", "cand_id", "source", *FEATURES, "core_key", "nums_key"]


def part_groups(cand_dir):
    """{(country, start): [part paths]} for s<source>_<country>_<key>_<start>.parquet."""
    groups = defaultdict(list)
    for p in sorted(cand_dir.glob("s*_*.parquet")):
        source, rest = p.stem.split("_", 1)
        country, _key, start = rest.rsplit("_", 2)
        groups[(country, start)].append(p)
    return groups


def finished(paths, n_sources=2, settle_s=60):
    return len(paths) == n_sources and all(time.time() - p.stat().st_mtime > settle_s for p in paths)


def main():
    FEATS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tables = None
    for (country, start), paths in sorted(part_groups(full_dir("test")).items()):
        done_flag = FEATS_DIR / f"{country}_{start}.done"
        if done_flag.exists() or not finished(paths):
            continue
        if tables is None:
            tables = Tables("test")
        cands = pl.concat([pl.read_parquet(p) for p in paths])
        for j, chunk in enumerate(iter_chunks(cands, 50_000)):
            f = pair_features(attach_text(chunk, tables)).with_columns(
                pl.col("core_n_b").alias("core_key"),
                pl.col("addr_n_b").str.extract_all(r"\d+").list.unique().list.sort()
                  .list.join(" ").alias("nums_key"))
            f.select(KEEP).with_columns(pl.col(FEATURES).cast(pl.Float32)).write_parquet(
                FEATS_DIR / f"{country}_{start}_{j:02d}.parquet")
        done_flag.touch()
        print(f"  {country} {start}: {cands.height:,} rows ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
