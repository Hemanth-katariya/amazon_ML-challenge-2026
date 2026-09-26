# Business Entity Resolution — ML Challenge 2026

For every Source-1 business, find its matching records in Source 2 and Source 3.

Pipeline: **normalize → block (TF-IDF top-k over name+address) → pair features →
LightGBM → per-entity decision rule**. CPU only; no external data, APIs or pretrained
models (LightGBM, scikit-learn and rapidfuzz are MIT/BSD licensed).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# .venv/bin/python -m pip install -r requirements.txt       # Linux / macOS
```

Paths live in `src/config.py` and can be overridden with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `ER_RAW_DIR` | folder with the organizers' `train/` and `test/` TSVs | local dataset folder |
| `ER_PARQUET_DIR` | parquet copies of the TSVs | `C:\ml2026\data\parquet` |
| `ER_WORK_DIR` | intermediate artefacts (candidates, features, model) | `C:\ml2026\data\work` |
| `ER_OUTPUT_DIR` | final submission files | `output/` next to `src/` |

## Reproduce end to end

Run from `src/`. Times are for a 10-core laptop with 16 GB RAM.

| Step | Command | Time | Produces |
|---|---|---|---|
| 1. TSV → parquet | `python convert_to_parquet.py` | 1 min | `ER_PARQUET_DIR/*.parquet` |
| 2. Transliteration dictionary | `python normalize.py` | 2 min | `translit_dict.json` (learned from train pairs only) |
| 3. Normalize text | `python normalize_all.py train test` | 9 min | `norm_*.parquet` |
| 4. Dev candidates | `python make_candidates.py dev` | 21 min | `cands_dev.parquet` (labelled) |
| 5. Full-density blocking, train | `python full_blocking.py train` | ~3.7 h | `full_train/*.parquet` |
| 6. Train matcher | `python train_matcher.py --rev` | ~25 min | `lgbm.txt`, `decision.json` |
| 7. Full-density blocking, test | `python full_blocking.py test` | ~3 h | `full_test/*.parquet` (= test candidates) |
| 8. Predict test | `python predict_test.py` | ~1 h | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Steps 5 and 7 are restartable: finished parts are skipped.

Validate the output with the organizers' checker (run from `student_resource/`):

```bash
python utils/validate_submission.py --matching <output>/matching_results.tsv \
    --candidate <output>/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Source map

| File | Role |
|---|---|
| `config.py` | paths |
| `convert_to_parquet.py` | raw TSV → parquet (quoting disabled; every column a string) |
| `normalize.py` | transliteration (learned dictionary + IAST fallback), accent/junk stripping, abbreviation canonicalization |
| `normalize_all.py` | multiprocess normalization of every source, cached |
| `blocking.py` | TF-IDF top-k index over `name + address`, per country and source |
| `make_candidates.py` / `full_blocking.py` | candidate generation (dev sample / every S1 entity) |
| `features.py` | pair features (name, address, house numbers, rarity, context) |
| `reverse.py` | candidate-side features from full-density blocking |
| `train_matcher.py` | LightGBM training, decision-rule selection on validation |
| `decide.py` | decision rules: threshold, two thresholds, expected-F0.5, one owner per record |
| `predict_test.py` | test scoring and submission files |
| `metric.py`, `test_metric.py` | local macro F0.5 (checked against the problem statement's example) |
| `split.py` | stable 10% validation split by S1 entity; country holdout |
| `eval_blocking.py`, `baselines.py` | blocking recall and reference scores |
