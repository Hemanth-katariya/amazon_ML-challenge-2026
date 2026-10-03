# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** kasukabe defence group  
**Members:** Abhinav Sudini, Nishanth Kumar Nelanti, Katariya Hemanth Kumar  
**Institute:** Indian Institute of Technology (IIT) Patna  
**Validation F0.5:** 0.9803 (test-like) · 0.9786 (plain) · **LB: 0.95**

---

## Problem

Given ~2.2M Source-1 businesses, find all matching records for each in Source 2 (~5.3M records) and Source 3 (~5.0M records). Evaluated with **macro F0.5** — precision-weighted, one score per Source-1 entity.

Key challenges:
- All-pairs comparison (2.2M × 10.3M) is infeasible — blocking is essential
- Indian-script names (Devanagari, Tamil, Telugu, Kannada) share no characters with Latin
- Test set carries ~2.3 unmatched decoys per entity vs 1.2 in training, grouped into near-miss "branches"
- France (15% of test) is absent from training data

---

## Solution Overview

```
Raw TSVs
   ↓ normalize.py / normalize_all.py
Normalized text  (transliteration, accent stripping, abbrev canonicalization)
   ↓ blocking.py / full_blocking.py
~40 candidates per S1 entity  (TF-IDF top-k over name + address)
   ↓ features.py / reverse.py
Pair features + reverse features  (name, address, house numbers, rarity, context)
   ↓ stack.py  (with augment.py)
Two-stage LightGBM  (stage 2 sees stage-1 probabilities across the entity's candidates)
   ↓ decide.py / predict_test.py
Per-entity gated expected-F0.5 decision rule
   ↓
matching_results.tsv + candidate_pairs.tsv
```

---

## Key Ideas

### 1. Transliteration learned from training pairs
A 1,347-entry dictionary is built by aligning `(Latin S1 name, Indic S2/S3 name)` position by position. Covers 100% of validation tokens and 96.4% of test tokens; unseen words fall back to rule-based IAST mapping.

### 2. Blocking over name + address as one document
A single TF-IDF index per `(country, source)` over `name + address` tokens. Top 20 candidates per source. Rare words like address fragments dominate naturally without any weighting. Recall: **98.1%** of true pairs appear in the top 40 candidates (ceiling F0.5 = 0.9935).

### 3. Reverse features
Every Source-1 entity is blocked, so for each candidate record we know: how many S1 entities retrieved it, the best competing score, whether this entity is its best match, and the margin. This halved false-positive merges (1,577 → 764 wrong pairs).

### 4. Two-stage LightGBM
- **Stage 1:** pair features + reverse features
- **Stage 2:** adds group summaries of stage-1 probabilities (rank within entity, probability-weighted support from candidates sharing house numbers or name, best probability in the other source)
- Stage-2 training uses 3-fold out-of-fold stage-1 probabilities to avoid leakage

### 5. Decoy-branch augmentation
The test set's decoy density (2.3/entity) is ~2× that of training (1.2/entity), and extra decoys arrive in *branches* — two or three records at a near-miss address that support each other. We reproduce this by cloning half of each entity's pure-decoy candidates into the other source and recomputing group features. This fixed the 0.03 gap between validation and leaderboard:

| Model | Plain val | Test-like val | LB |
|---|---|---|---|
| Before augmentation | 0.9792 | 0.9504 | 0.95 |
| After augmentation | 0.9786 | **0.9803** | submitted |

### 6. Gated decision rule
An entity gets any match only if its best candidate clears a strict threshold (0.60), protecting singletons. The match count is then chosen to maximize expected F0.5 over the probability-sorted candidates.

---

## Results

| Ablation step | Val F0.5 |
|---|---|
| Predict "no match" everywhere | 0.0557 |
| Stage 1, base features | 0.9706 |
| + house-number / rarity / missing-word features | 0.9739 |
| + group agreement + gated rule | 0.9755 |
| + stage 2 | 0.9775 |
| + reverse features | **0.9792** |
| Blocking ceiling (perfect matcher) | 0.9935 |

---

## Repository Structure

```
code/
  business_entity_resolution/
    src/              # full pipeline (see source map below)
    README.md         # step-by-step reproduction with timings
    requirements.txt  # pinned dependencies
Documentation_template.md   # full methodology write-up
```

### Source Map

| File | Role |
|---|---|
| `config.py` | paths and environment variable overrides |
| `convert_to_parquet.py` | raw TSV → parquet |
| `normalize.py` | build transliteration dictionary from training pairs |
| `normalize_all.py` | multiprocess normalization for every source |
| `blocking.py` | TF-IDF top-k index over name + address |
| `make_candidates.py` | candidate generation for a dev sample |
| `full_blocking.py` | full-density candidate generation (train / test) |
| `features.py` | pair features (name, address, house numbers, rarity, context) |
| `reverse.py` | reverse features from full-density blocking |
| `augment.py` | decoy-branch augmentation |
| `stack.py` | two-stage LightGBM with augmentation |
| `train_matcher.py` | dev feature cache, single-stage baseline |
| `test_features.py` | test pair features (restartable) |
| `decide.py` | decision rules (threshold, gated expected-F0.5) |
| `predict_test.py` | test scoring and submission output |
| `metric.py` / `test_metric.py` | local macro F0.5 |
| `split.py` | stable 10% validation split by S1 entity |
| `eval_blocking.py` / `baselines.py` | blocking recall and baselines |

---

## Setup

Python 3.12 (developed on 3.12.5, Windows 11; nothing is Windows-specific).

```bash
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -r code/business_entity_resolution/requirements.txt

# Linux / macOS
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

Place the organizers' dataset folder (containing `train/` and `test/`) next to `code/`, or set `ER_RAW_DIR` to point at it.

| Variable | Meaning | Default |
|---|---|---|
| `ER_RAW_DIR` | organizers' `train/` and `test/` TSVs | `<root>/dataset` |
| `ER_PARQUET_DIR` | parquet copies of TSVs | `<root>/data/parquet` |
| `ER_WORK_DIR` | intermediate artefacts | `<root>/data/work` |
| `ER_OUTPUT_DIR` | submission output files | `output/` next to `src/` |

Needs ~10 GB disk for intermediate artefacts; developed on a 16 GB machine.

---

## Reproduce End-to-End

Run from `code/business_entity_resolution/src/`. Timings for a 10-core laptop, 16 GB RAM.

| Step | Command | Time |
|---|---|---|
| 1. TSV → parquet | `python convert_to_parquet.py` | 1 min |
| 2. Build transliteration dict | `python normalize.py` | 2 min |
| 3. Normalize all text | `python normalize_all.py train test` | 9 min |
| 4. Dev candidates | `python make_candidates.py dev` | 21 min |
| 5. Full blocking (train) | `python full_blocking.py train` | ~3.7 h |
| 6. Dev pair features | `python train_matcher.py --rev` | ~40 min |
| 7. Two-stage model + augmentation | `python stack.py --rev --aug=0.5` | ~55 min |
| 8. Full blocking (test) | `python full_blocking.py test` | ~3 h |
| 9. Test pair features | `python test_features.py` | ~2.5 h |
| 10. Predict & write submission | `python predict_test.py` | ~1.4 h |

Steps 5, 8 and 9 are restartable; finished chunks are skipped automatically.

Validate the output:
```bash
python utils/validate_submission.py \
    --matching <output>/matching_results.tsv \
    --candidate <output>/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

---

## Dependencies

| Library | Version | License |
|---|---|---|
| lightgbm | 4.7.0 | MIT |
| polars | 1.44.2 | MIT |
| scikit-learn | 1.9.1 | BSD |
| scipy | 1.18.1 | BSD |
| numpy | 2.5.3 | BSD |
| RapidFuzz | 3.14.6 | MIT |
| sparse-dot-topn | 1.2.0 | Apache-2.0 |
| indic_transliteration | 2.3.82 | MIT |
| anyascii | 0.3.3 | ISC |
| pyarrow | 25.0.1 | Apache-2.0 |

No pretrained language models. No external data, APIs or lookups. All models and dictionaries are derived from the provided training and test files only.

---

## Compliance

- **No external data:** every model, dictionary and statistic is derived from the provided training and test files.
- **Test files:** used only for unlabelled counts (records per entity); never used for labels or any form of supervised signal.
- **No pretrained models:** LightGBM gradient-boosted trees only (far below 8B parameters).
