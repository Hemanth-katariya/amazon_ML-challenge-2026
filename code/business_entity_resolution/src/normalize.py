"""Text normalization for business names and addresses.

Pipeline for a raw string:
  1. Indic-script tokens -> Latin (dictionary learned from training pairs, with a
     rule-based IAST fallback for unseen words).
  2. Unicode NFKD + strip accents, lowercase.
  3. Remove junk (null markers, bracket noise), split on non-alphanumerics.
  4. Canonicalize abbreviations (street types, legal forms, state names).
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

import polars as pl
from anyascii import anyascii
from indic_transliteration import sanscript
from indic_transliteration.detect import detect

import config

INDIC_RE = re.compile(r"[ऀ-෿]")
INDIC_PATTERN = r"[ऀ-෿]"  # same range, for polars expressions
TRANSLIT_DICT_PATH = config.WORK_DIR / "translit_dict.json"

_SCHEMES = {name: getattr(sanscript, name.upper()) for name in (
    "devanagari", "tamil", "telugu", "kannada", "bengali",
    "gujarati", "gurmukhi", "malayalam", "oriya")}

# The only Indic tokens that occur in addresses are state names (17 distinct).
INDIC_STATES = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर": "uttar", "प्रदेश": "pradesh",
    "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "tg",
    "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "ગુજરાત": "gj", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od", "ఆంధ్ర": "andhra",
    "मध्य": "madhya",
}

# Token-level canonical forms. Keys are already lowercase and accent-free.
ADDRESS_CANON = {
    # street types
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd",
    "court": "ct", "place": "pl", "square": "sq", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "trail": "trl", "circle": "cir", "route": "rte", "chemin": "ch",
    "allee": "all", "impasse": "imp",
    # unit / building
    "apartment": "apt", "suite": "ste", "floor": "fl", "building": "bldg",
    "number": "no", "nos": "no", "house": "h",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "nord": "n", "sud": "s", "est": "e", "ouest": "w",
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn_us",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "wisconsin": "wi", "wyoming": "wy",
    # Indian states (single-token forms)
    "maharashtra": "mh", "delhi": "dl", "karnataka": "ka", "telangana": "tg",
    "haryana": "hr", "rajasthan": "rj", "kerala": "kl", "bihar": "br", "gujarat": "gj",
    "punjab": "pb", "odisha": "od", "orissa": "od",
}
# Multi-word state names collapse to one token before the token-level map.
PHRASE_CANON = [
    (re.compile(r"\bnew (hampshire|jersey|mexico|york)\b"), lambda m: "n" + m.group(1)[0]),
    (re.compile(r"\bnorth carolina\b"), lambda m: "nc"),
    (re.compile(r"\bsouth carolina\b"), lambda m: "sc"),
    (re.compile(r"\bnorth dakota\b"), lambda m: "nd"),
    (re.compile(r"\bsouth dakota\b"), lambda m: "sd"),
    (re.compile(r"\bwest virginia\b"), lambda m: "wv"),
    (re.compile(r"\brhode island\b"), lambda m: "ri"),
    (re.compile(r"\btamil nadu\b"), lambda m: "tn"),
    (re.compile(r"\buttar pradesh\b"), lambda m: "up"),
    (re.compile(r"\bmadhya pradesh\b"), lambda m: "mp"),
    (re.compile(r"\bandhra pradesh\b"), lambda m: "ap"),
    (re.compile(r"\bwest bengal\b"), lambda m: "wb"),
]

LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private",
    "corp", "corporation", "co", "company", "plc", "pllc", "pc", "dba",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop",
}
GENERIC = {"services", "center", "centre", "partners", "group", "the", "and", "et", "of"}

JUNK_RE = re.compile(r"<null>|\bnull\b|\bnan\b|\bnone\b|\bn/a\b")
PHONE_RE = re.compile(r"\d{7,}")  # phone numbers pasted into names
LEADING_ZEROS_RE = re.compile(r"\b0+(?=\d)")  # "0112" -> "112" (house-number padding)
# Digit-for-letter noise inside name words: "0xford", "6ulf", "8one".
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b"})
MIXED_RE = re.compile(r"(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[0-9])[a-z0-9]+")


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def split_tokens(s: str):
    """Split on punctuation/symbols/whitespace only (keeps Indic combining marks)."""
    s = unicodedata.normalize("NFC", s).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in s).split()


def _latin_plain(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _strip_accents(s).lower()).strip()


@lru_cache(maxsize=None)
def rule_translit(tok: str) -> str:
    try:
        scheme = _SCHEMES.get(detect(tok))
        out = sanscript.transliterate(tok, scheme, sanscript.IAST) if scheme else anyascii(tok)
    except Exception:
        out = anyascii(tok)
    out = _latin_plain(out)
    return re.sub(r"(?<=[b-df-hj-np-tv-z])a\b", "", out)  # drop word-final inherent 'a'


# ---------------------------------------------------------------- dictionary

def learn_translit_dict(pairs: pl.DataFrame, min_votes: int = 2, min_share: float = 0.5):
    """Learn Indic-token -> Latin-token from (Latin S1 name, Indic S2/S3 name) pairs.

    Pairs with equal token counts are aligned position by position; each Indic token
    takes its most frequent Latin counterpart if it is a clear majority.
    """
    votes = defaultdict(Counter)
    for latin_name, indic_name in zip(pairs["n1"], pairs["n2"]):
        a, b = split_tokens(_latin_plain(latin_name)), split_tokens(indic_name)
        if len(a) != len(b):
            continue
        for w, t in zip(a, b):
            if INDIC_RE.search(t):
                votes[t][w] += 1
    out = {}
    for t, c in votes.items():
        w, n = c.most_common(1)[0]
        if n >= min_votes and n / sum(c.values()) >= min_share:
            out[t] = w
    return out


def build_translit_dict(train_s1_ids=None) -> dict:
    """Learn the dictionary from training ground truth; optionally restrict to some S1 ids."""
    gt = pl.read_parquet(config.ground_truth_parquet())
    if train_s1_ids is not None:
        gt = gt.filter(pl.col("source1_entity_id").is_in(list(train_s1_ids)))
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids", empty_as_null=True).drop_nulls())
    indic = (pl.concat([pl.read_parquet(config.parquet_path("train", s)) for s in (2, 3)])
               .filter(pl.col("business_name").str.contains(INDIC_PATTERN))
               .select("entity_id", pl.col("business_name").alias("n2")))
    s1 = pl.read_parquet(config.parquet_path("train", 1)).select(
        "entity_id", pl.col("business_name").alias("n1"))
    pairs = (pairs.join(indic, left_on="matched_entity_ids", right_on="entity_id")
                  .join(s1, left_on="source1_entity_id", right_on="entity_id"))
    return learn_translit_dict(pairs)


def save_translit_dict(d: dict, path=TRANSLIT_DICT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")


def load_translit_dict(path=TRANSLIT_DICT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- normalizers

class Normalizer:
    def __init__(self, translit: dict):
        self.translit = translit

    def _indic_token(self, tok: str, is_address: bool) -> str:
        if is_address and tok in INDIC_STATES:
            return INDIC_STATES[tok]
        return self.translit.get(tok) or rule_translit(tok)

    def _prepare(self, s: str, is_address: bool) -> str:
        if not s:
            return ""
        if INDIC_RE.search(s):
            s = " ".join(self._indic_token(t, is_address) if INDIC_RE.search(t) else t
                         for t in split_tokens(s))
        s = _strip_accents(s).lower()
        s = JUNK_RE.sub(" ", s).replace("&", " and ")
        return re.sub(r"[^a-z0-9]+", " ", s).strip()

    def address(self, s: str) -> str:
        s = LEADING_ZEROS_RE.sub("", self._prepare(s, is_address=True))
        for pattern, repl in PHRASE_CANON:
            s = pattern.sub(repl, s)
        return " ".join(ADDRESS_CANON.get(t, t) for t in s.split())

    def name(self, s: str) -> str:
        s = self._prepare(s, is_address=False)
        toks = [t.translate(LEET) if MIXED_RE.fullmatch(t) else t
                for t in s.split() if not PHONE_RE.fullmatch(t)]
        # collapse immediate duplicates ("investments investments", "ap ap")
        toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
        return " ".join(toks)

    @staticmethod
    def core_name(name_norm: str) -> str:
        """Name without legal forms / generic words; falls back to the full name."""
        toks = [t for t in name_norm.split() if t not in LEGAL and t not in GENERIC]
        return " ".join(toks) if toks else name_norm


def normalize_frame(df: pl.DataFrame, norm: Normalizer) -> pl.DataFrame:
    """Add name_n, core_n, addr_n columns to a source frame."""
    names = [norm.name(s) for s in df["business_name"]]
    addrs = [norm.address(s) for s in df["business_address"]]
    return df.with_columns(
        pl.Series("name_n", names),
        pl.Series("core_n", [norm.core_name(n) for n in names]),
        pl.Series("addr_n", addrs),
    )


if __name__ == "__main__":
    from split import random_split, s1_frame
    train_ids, _ = random_split(s1_frame())
    d = build_translit_dict(train_ids)   # learned only from the train side of the split
    save_translit_dict(d)
    print(f"saved {len(d)} entries -> {TRANSLIT_DICT_PATH}")
    n = Normalizer(d)
    for s in ["राम मार्केटिंग प्राइवेट लिमिटेड", "Payne Énterprises", "Obsidian, [[LLC]]",
              "AP AP Hospitality Incorporated", "Chordia + Pagnters - 7306204978"]:
        print(f"  name  {s!r} -> {n.name(s)!r} | core {n.core_name(n.name(s))!r}")
    for s in ["KANSAS CITY, MO, 630 45ND TERRACE, null",
              "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு",
              "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
              "33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA", "Fremont St, Peoria, Illinois"]:
        print(f"  addr  {s!r} -> {n.address(s)!r}")
