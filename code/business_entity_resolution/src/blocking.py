"""Candidate generation: TF-IDF top-k search per country, per target source.

Primary key is name + address in ONE document ("both_n"). Names repeat across many
distinct businesses (36-44% of S1 entities share their exact name with another one),
so a name-only search drowns in namesakes and an address-only search in co-located
businesses; scoring them together ranks the true match far higher (diag: 97% of true
pairs in the top 10 vs 83% for address alone).
"""
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

import config
from normalize_all import norm_path


@dataclass(frozen=True)
class Key:
    name: str      # label stored in the candidate frame
    column: str    # normalized text column to index
    analyzer: str  # "word" or "char_wb"
    ngram: tuple
    k: int
    max_df: int    # absolute document-frequency cap


NO_CAP = 10**9

# max_df=200k drops only ~14 hyper-common tokens (private, limited, state codes):
# queries get 2-2.5x faster for ~0.1pt recall (cap sweep on 3k val queries/country).
DEFAULT_KEYS = (
    Key("both", "both_n", "word", (1, 1), k=20, max_df=200_000),
)


def _vectorizer(key: Key) -> TfidfVectorizer:
    kw = dict(analyzer=key.analyzer, ngram_range=key.ngram, min_df=2, max_df=key.max_df,
              lowercase=False, sublinear_tf=True, dtype=np.float32)
    if key.analyzer == "word":
        kw["token_pattern"] = r"\S+"
    return TfidfVectorizer(**kw)


class Index:
    """TF-IDF index over one corpus for one key. Fitting is the expensive part, so an
    index is built once and queried many times (validation, training sample, test)."""

    def __init__(self, corpus: pl.DataFrame, key: Key):
        self.key = key
        self.vec = _vectorizer(key)
        self.C = self.vec.fit_transform(corpus[key.column].to_list())
        self.ids = corpus["entity_id"].to_numpy()

    def query(self, queries: pl.DataFrame, n_threads: int = -1) -> pl.DataFrame:
        """Top-k corpus rows per query row -> frame(s1_id, cand_id, key, score, rank)."""
        Q = self.vec.transform(queries[self.key.column].to_list())
        R = sp_matmul_topn(Q, self.C, top_n=self.key.k, sort=True, n_threads=n_threads)
        counts = np.diff(R.indptr)
        # rank within each query row: position minus the row's start offset
        rank = (np.arange(R.nnz) - np.repeat(R.indptr[:-1], counts)).astype(np.int16)
        return pl.DataFrame({
            "s1_id": np.repeat(queries["entity_id"].to_numpy(), counts),
            "cand_id": self.ids[R.indices],
            "key": self.key.name,
            "score": R.data.astype(np.float32),
            "rank": rank,
        })


def topk(queries: pl.DataFrame, corpus: pl.DataFrame, key: Key, n_threads: int = -1):
    return Index(corpus, key).query(queries, n_threads)


def load_norm(split: str, source: int) -> pl.DataFrame:
    return pl.read_parquet(norm_path(split, source)).with_columns(
        (pl.col("name_n") + " " + pl.col("addr_n")).alias("both_n"))


def generate_to_dir(split: str, out_dir, keys=DEFAULT_KEYS, chunk: int = 200_000):
    """Candidates for EVERY S1 entity of `split`, streamed to parquet parts in `out_dir`.

    Memory stays bounded: one corpus index at a time, queries in chunks. Each part is
    written as soon as it is computed, and parts already on disk are skipped, so an
    interrupted run can simply be restarted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["entity_id", "country", "both_n"]
    s1 = load_norm(split, 1).select(cols)
    for source in (2, 3):
        other = load_norm(split, source).select(cols)
        for country in sorted(s1["country"].unique().to_list()):
            q_all = s1.filter(pl.col("country") == country)
            c = other.filter(pl.col("country") == country)
            for key in keys:
                parts = [out_dir / f"s{source}_{country}_{key.name}_{i:04d}.parquet"
                         for i in range(0, q_all.height, chunk)]
                if all(p.exists() for p in parts):
                    continue
                t0 = time.time()
                index = Index(c, key)
                print(f"    S{source} {country:<7} {key.name} fit {time.time() - t0:.0f}s "
                      f"({c.height:,} docs), {q_all.height:,} queries", flush=True)
                for path, start in zip(parts, range(0, q_all.height, chunk)):
                    if path.exists():
                        continue
                    t1 = time.time()
                    (index.query(q_all.slice(start, chunk))
                          .with_columns(pl.lit(source, pl.Int8).alias("source"))
                          .write_parquet(path))
                    print(f"      {path.name} {time.time() - t1:.0f}s", flush=True)
                del index


def generate(split: str, s1_ids=None, keys=DEFAULT_KEYS, verbose=True) -> pl.DataFrame:
    """Candidates for the S1 entities of `split` (optionally only `s1_ids`)."""
    s1 = load_norm(split, 1)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(list(s1_ids)))
    parts = []
    for source in (2, 3):
        other = load_norm(split, source)
        for country in sorted(s1["country"].unique().to_list()):
            q = s1.filter(pl.col("country") == country)
            c = other.filter(pl.col("country") == country)
            for key in keys:
                t0 = time.time()
                index = Index(c, key)
                t1 = time.time()
                parts.append(index.query(q).with_columns(pl.lit(source, pl.Int8).alias("source")))
                if verbose:
                    print(f"    S{source} {country:<7} {key.name:<5} q={q.height:,} "
                          f"corpus={c.height:,} fit={t1 - t0:.0f}s query={time.time() - t1:.0f}s",
                          flush=True)
    return pl.concat(parts)
