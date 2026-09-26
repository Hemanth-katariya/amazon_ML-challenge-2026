"""Normalize every source file once (multiprocess) and cache the result as parquet.

Output columns: entity_id, country, name_n, core_n, addr_n
"""
import os
import sys
import time
from multiprocessing import Pool

import polars as pl

import config
from normalize import Normalizer, load_translit_dict

CHUNK = 50_000
_norm = None


def _init():
    global _norm
    _norm = Normalizer(load_translit_dict())


def _work(chunk):
    names, addrs = chunk
    name_n = [_norm.name(s) for s in names]
    return name_n, [_norm.core_name(n) for n in name_n], [_norm.address(s) for s in addrs]


def norm_path(split: str, source: int):
    return config.WORK_DIR / f"norm_{split}_source{source}.parquet"


def normalize_source(split: str, source: int, pool: Pool):
    t0 = time.time()
    df = pl.read_parquet(config.parquet_path(split, source))
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    chunks = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]
    name_n, core_n, addr_n = [], [], []
    for a, b, c in pool.imap(_work, chunks):
        name_n += a
        core_n += b
        addr_n += c
    out = df.select("entity_id", "country").with_columns(
        pl.Series("name_n", name_n), pl.Series("core_n", core_n), pl.Series("addr_n", addr_n))
    out.write_parquet(norm_path(split, source))
    print(f"  {split} source{source}: {out.height:,} rows in {time.time() - t0:.0f}s")


def main(splits=("train", "test")):
    workers = max(1, (os.cpu_count() or 2) - 2)
    with Pool(workers, initializer=_init) as pool:
        for split in splits:
            for source in (1, 2, 3):
                normalize_source(split, source, pool)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("train", "test"))
