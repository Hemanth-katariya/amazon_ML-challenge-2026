"""Local replica of the leaderboard metric: macro F_0.5 over Source-1 entities.

Rules (from the problem statement):
  * F_0.5 = 1.25*P*R / (0.25*P + R), computed per Source-1 entity, then averaged.
  * True singleton (no matches): 1.0 if prediction is empty, else 0.0.
  * Entity with matches but empty / fully wrong prediction: 0.0.
  * An entity missing from the prediction counts as an empty prediction.
"""
from typing import Dict, Iterable, Mapping, Set

import polars as pl

BETA2 = 0.25  # beta = 0.5


def parse_id_list(s: str) -> Set[str]:
    return {x for x in s.split(",") if x} if s else set()


def f05(pred: Set[str], true: Set[str]) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
              ids: Iterable[str] = None) -> float:
    """Average F_0.5 over `ids` (default: every entity in `truth`)."""
    ids = list(truth) if ids is None else list(ids)
    return sum(f05(pred.get(i, set()), truth[i]) for i in ids) / len(ids)


def score_report(pred: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
                 ids: Iterable[str] = None) -> Dict[str, float]:
    """Overall score plus the breakdown that tells us *where* points are lost."""
    ids = list(truth) if ids is None else list(ids)
    single = [i for i in ids if not truth[i]]
    multi = [i for i in ids if truth[i]]
    prec, rec = [], []
    for i in multi:
        p, t = pred.get(i, set()), truth[i]
        tp = len(p & t)
        prec.append(tp / len(p) if p else 0.0)
        rec.append(tp / len(t))
    return {
        "f05": macro_f05(pred, truth, ids),
        "n": len(ids),
        "singleton_share": len(single) / len(ids),
        "singleton_f05": macro_f05(pred, truth, single) if single else float("nan"),
        "matched_f05": macro_f05(pred, truth, multi) if multi else float("nan"),
        "matched_precision": sum(prec) / len(prec) if prec else float("nan"),
        "matched_recall": sum(rec) / len(rec) if rec else float("nan"),
    }


def read_id_list_tsv(path, id_col: str, list_col: str) -> Dict[str, Set[str]]:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    df = df.with_columns(pl.col(list_col).fill_null(""))
    return {k: parse_id_list(v) for k, v in zip(df[id_col], df[list_col])}


def load_ground_truth(path) -> Dict[str, Set[str]]:
    df = pl.read_parquet(path)
    return {k: parse_id_list(v)
            for k, v in zip(df["source1_entity_id"], df["matched_entity_ids"])}
