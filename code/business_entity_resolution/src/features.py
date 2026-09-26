"""Pair features for (S1 record, candidate) rows produced by blocking.

Country is deliberately NOT a feature: France only appears in test.
Feature computation is row-local except the per-S1 context features, so callers
must pass whole S1 groups (all candidates of an S1 entity) in one chunk.
"""
import math
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config
from blocking import load_norm
from normalize import INDIC_PATTERN

NUM_RE = re.compile(r"\d+")
POSTCODE_RE = re.compile(r"^\d{5,6}$")  # US ZIP / France code postal (5), India PIN (6)

FEATURES = [
    "source", "score", "rank", "score_rel", "score_gap",
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_exact",
    "c_ratio", "c_jw", "c_tset", "c_jacc", "c_first_eq", "c_ntok_a", "c_ntok_b",
    "a_ratio", "a_tset", "a_partial", "a_jacc", "a_empty_b",
    "num_shared", "num_jacc", "num_first_eq", "num_best_sim", "a_has_num", "b_has_num",
    "zip_match", "zip_conflict",
    "b_indic", "name_freq", "addr_freq",
    "n_tset_gap", "a_tset_gap", "c_ratio_gap",
    # decoy discrimination (siblings: near-identical name / same street, other number)
    "c_missing_frac", "c_extra_n", "num_a_in_b", "num_conflict", "num_trunc",
    "num_absdiff_log", "a_alpha_jacc", "cand_name_freq", "unique_exact",
]


# ---------------------------------------------------------------- inputs

def rarity_tables(split: str):
    """How many S1 entities (same country) share each core name / each address, and how
    many S2/S3 records share each core name."""
    s1 = load_norm(split, 1)
    name = s1.group_by("country", "core_n").agg(pl.len().alias("name_freq"))
    addr = s1.group_by("country", "addr_n").agg(pl.len().alias("addr_freq"))
    cand = (pl.concat([load_norm(split, s).select("country", "core_n") for s in (2, 3)])
              .group_by("country", "core_n").agg(pl.len().alias("cand_name_freq")))
    return name, addr, cand


def attach_text(cands: pl.DataFrame, split: str) -> pl.DataFrame:
    """Join normalized S1 and candidate text plus rarity counts onto candidate rows."""
    cols = ["entity_id", "country", "name_n", "core_n", "addr_n"]
    s1 = load_norm(split, 1).select(cols).filter(
        pl.col("entity_id").is_in(cands["s1_id"].unique().implode()))
    wanted = cands["cand_id"].unique().implode()
    other = pl.concat([
        load_norm(split, s).select(cols[0], *cols[2:]).filter(pl.col("entity_id").is_in(wanted))
        for s in (2, 3)])
    indic = pl.concat([
        pl.read_parquet(config.parquet_path(split, s), columns=["entity_id", "business_name"])
          .filter(pl.col("entity_id").is_in(wanted))
          .select("entity_id", pl.col("business_name").str.contains(INDIC_PATTERN)
                  .cast(pl.Int8).alias("b_indic"))
        for s in (2, 3)])
    name_freq, addr_freq, cand_freq = rarity_tables(split)
    return (cands
            .join(s1.rename({c: f"{c}_a" for c in cols[2:]}).rename({"entity_id": "s1_id"}),
                  on="s1_id")
            .join(other.rename({c: f"{c}_b" for c in cols[2:]}).rename({"entity_id": "cand_id"}),
                  on="cand_id")
            .join(indic.rename({"entity_id": "cand_id"}), on="cand_id")
            .join(name_freq.rename({"core_n": "core_n_a"}), on=["country", "core_n_a"], how="left")
            .join(addr_freq.rename({"addr_n": "addr_n_a"}), on=["country", "addr_n_a"], how="left")
            .join(cand_freq.rename({"core_n": "core_n_b"}), on=["country", "core_n_b"], how="left"))


# ---------------------------------------------------------------- features

def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _jacc(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if x or y else 0.0


def _token_present(tok: str, other_toks: list, other_joined: str) -> bool:
    """Exact, glued ("solutionstrinityenergy") or typo'd (ratio >= 80) presence."""
    if tok in other_toks or (len(tok) >= 4 and tok in other_joined):
        return True
    return process.extractOne(tok, other_toks, scorer=fuzz.ratio, score_cutoff=80) is not None


def _set_features(core_a, core_b, name_b, addr_a, addr_b):
    """Token-set and number features that need Python sets (one pass over rows)."""
    n = len(core_a)
    out = {k: np.zeros(n, np.float32) for k in (
        "c_jacc", "c_first_eq", "c_ntok_a", "c_ntok_b", "a_jacc", "num_shared", "num_jacc",
        "num_first_eq", "num_best_sim", "a_has_num", "b_has_num", "zip_match", "zip_conflict",
        "c_missing_frac", "c_extra_n", "num_a_in_b", "num_conflict", "num_trunc",
        "num_absdiff_log", "a_alpha_jacc")}
    for i in range(n):
        ca, cb = core_a[i].split(), core_b[i].split()
        out["c_jacc"][i] = _jacc(set(ca), set(cb))
        out["c_first_eq"][i] = bool(ca and cb and ca[0] == cb[0])
        out["c_ntok_a"][i], out["c_ntok_b"][i] = len(ca), len(cb)
        # distinctive S1 words missing from the candidate / unexplained extra words
        nb_toks = name_b[i].split()
        nb_joined = name_b[i].replace(" ", "")
        if ca:
            missing = sum(not _token_present(t, nb_toks, nb_joined) for t in ca)
            out["c_missing_frac"][i] = missing / len(ca)
        ca_joined = core_a[i].replace(" ", "")
        out["c_extra_n"][i] = sum(not _token_present(t, ca, ca_joined) for t in cb)

        ta, tb = addr_a[i].split(), addr_b[i].split()
        out["a_jacc"][i] = _jacc(set(ta), set(tb))
        out["a_alpha_jacc"][i] = _jacc({t for t in ta if not t.isdigit()},
                                       {t for t in tb if not t.isdigit()})
        na, nb = NUM_RE.findall(addr_a[i]), NUM_RE.findall(addr_b[i])
        out["a_has_num"][i], out["b_has_num"][i] = bool(na), bool(nb)
        if na and nb:
            sa, sb = set(na), set(nb)
            shared = sa & sb
            out["num_shared"][i] = len(shared)
            out["num_jacc"][i] = _jacc(sa, sb)
            out["num_first_eq"][i] = na[0] == nb[0]
            out["num_a_in_b"][i] = len(shared) / len(sa)
            # near-equal numbers: "1607" vs "16076", "56" vs "1056"
            out["num_best_sim"][i] = max(fuzz.ratio(x, y) for x in sa for y in sb)
            # truncation / padding noise: one number contained in another
            trunc = any(x != y and (x in y or y in x) for x in sa for y in sb)
            out["num_trunc"][i] = trunc
            # a real conflict: numbers present on both sides, none shared or nested
            out["num_conflict"][i] = not shared and not trunc
            # adjacent buildings (6850 vs 6855) have a small gap
            out["num_absdiff_log"][i] = math.log1p(min(abs(int(x) - int(y))
                                                       for x in sa for y in sb))
            za = {t for t in sa if POSTCODE_RE.match(t)}
            zb = {t for t in sb if POSTCODE_RE.match(t)}
            if za and zb:
                out["zip_match"][i] = bool(za & zb)
                out["zip_conflict"][i] = not (za & zb)
    return out


def pair_features(df: pl.DataFrame) -> pl.DataFrame:
    """df = attach_text output (whole S1 groups). Returns df plus FEATURES columns."""
    na, nb = df["name_n_a"].to_list(), df["name_n_b"].to_list()
    ca, cb = df["core_n_a"].to_list(), df["core_n_b"].to_list()
    aa, ab = df["addr_n_a"].to_list(), df["addr_n_b"].to_list()
    cols = {
        "n_ratio": _cp(fuzz.ratio, na, nb),
        "n_tsort": _cp(fuzz.token_sort_ratio, na, nb),
        "n_tset": _cp(fuzz.token_set_ratio, na, nb),
        "n_partial": _cp(fuzz.partial_ratio, na, nb),
        "c_ratio": _cp(fuzz.ratio, ca, cb),
        "c_jw": _cp(JaroWinkler.normalized_similarity, ca, cb),
        "c_tset": _cp(fuzz.token_set_ratio, ca, cb),
        "a_ratio": _cp(fuzz.ratio, aa, ab),
        "a_tset": _cp(fuzz.token_set_ratio, aa, ab),
        "a_partial": _cp(fuzz.partial_ratio, aa, ab),
    }
    cols.update(_set_features(ca, cb, nb, aa, ab))
    out = df.with_columns(
        *[pl.Series(k, v) for k, v in cols.items()],
        (pl.col("name_n_a") == pl.col("name_n_b")).cast(pl.Int8).alias("n_exact"),
        (pl.col("addr_n_b") == "").cast(pl.Int8).alias("a_empty_b"),
        # unique S1 name and exact core-name agreement: ~94-99% match rate even with
        # an empty address (vs ~2-9% when the name has namesakes)
        ((pl.col("name_freq").fill_null(0) <= 1) & (pl.col("core_n_a") == pl.col("core_n_b")))
        .cast(pl.Int8).alias("unique_exact"),
        pl.col("name_freq").fill_null(0).log1p().alias("name_freq"),
        pl.col("addr_freq").fill_null(0).log1p().alias("addr_freq"),
        pl.col("cand_name_freq").fill_null(0).log1p().alias("cand_name_freq"),
    )
    # context within the S1 entity's candidate list
    return out.with_columns(
        (pl.col("score") / pl.col("score").max().over("s1_id")).alias("score_rel"),
        (pl.col("score").max().over("s1_id") - pl.col("score")).alias("score_gap"),
        (pl.col("n_tset").max().over("s1_id") - pl.col("n_tset")).alias("n_tset_gap"),
        (pl.col("a_tset").max().over("s1_id") - pl.col("a_tset")).alias("a_tset_gap"),
        (pl.col("c_ratio").max().over("s1_id") - pl.col("c_ratio")).alias("c_ratio_gap"),
    )


def build(cands: pl.DataFrame, split: str, chunk_s1: int = 50_000) -> pl.DataFrame:
    """Features for all candidates, processed in chunks of whole S1 groups."""
    df = attach_text(cands, split)
    ids = df["s1_id"].unique().sort()
    parts = []
    for i in range(0, len(ids), chunk_s1):
        chunk = df.filter(pl.col("s1_id").is_in(ids[i:i + chunk_s1].implode()))
        parts.append(pair_features(chunk))
    return pl.concat(parts)
