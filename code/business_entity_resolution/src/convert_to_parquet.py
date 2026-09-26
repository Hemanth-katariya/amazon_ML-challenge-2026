"""One-time conversion of the raw TSVs to parquet.

Quoting is disabled on purpose: business names contain apostrophes and stray
double quotes, and the files are plain tab-separated with no quoting.
"""
import sys
import time

import polars as pl

import config


def read_tsv(path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,  # every column stays a string
    )
    # Empty fields (e.g. missing address, singleton match list) become "" not null.
    return df.with_columns(pl.all().fill_null(""))


def convert(src, dst) -> int:
    t0 = time.time()
    df = read_tsv(src)
    df.write_parquet(dst, compression="zstd")
    print(f"  {src.name}: {df.height:,} rows, cols={df.columns} "
          f"-> {dst.name} ({time.time() - t0:.1f}s)")
    return df.height


def main():
    config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for (split, source), rel in config.SOURCE_FILES.items():
        convert(config.RAW_DIR / rel, config.parquet_path(split, source))
    convert(config.RAW_DIR / config.GROUND_TRUTH_FILE, config.ground_truth_parquet())


if __name__ == "__main__":
    sys.exit(main())
