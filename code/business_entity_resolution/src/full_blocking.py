"""Block EVERY S1 entity of a split (restartable; parts land in WORK_DIR/full_<split>/).

Full density matters: reverse features ("which S1 entities retrieved this record, and
is this one the best?") and one-owner resolution only work when every competing S1
entity is present, as it is at test time.

  python full_blocking.py train
  python full_blocking.py test
"""
import sys
import time

import config
from blocking import generate_to_dir


def full_dir(split: str):
    return config.WORK_DIR / f"full_{split}"


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t0 = time.time()
    generate_to_dir(split, full_dir(split))
    print(f"done: {split} in {(time.time() - t0) / 60:.1f} min", flush=True)
