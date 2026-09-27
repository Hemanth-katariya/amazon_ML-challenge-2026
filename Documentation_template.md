# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Hemanth-katariya *(placeholder — replace before final submission)*
**Team Members:** Hemanth Katariya *(placeholder — replace before final submission)*
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve each Source-1 business to its Source-2/3 records with a four-step pipeline: script-aware text normalization, TF-IDF blocking over **name and address combined**, a two-stage LightGBM pair classifier, and a per-entity decision rule tuned directly for macro F0.5. The key ideas are a transliteration dictionary **learned from the training pairs** (Indian-script names), **reverse features** from blocking every Source-1 entity (does another Source-1 entity claim this record more strongly?), a second stage that reads the first stage's probabilities for the entity's other candidates, and **decoy-branch augmentation**: the test set carries twice the decoy records of the training set, grouped into mutually supporting "branches", so we simulate that density in training and select the decision rule on a test-like validation set. Validation macro F0.5: **0.9786** on the plain validation slice and **0.9803** on the test-like one (blocking ceiling 0.9935).

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training data before modelling:

| Finding | Number | Consequence |
|---|---|---|
| Scale | 2.2M S1 × 10.3M S2+S3 records (test: 1.73M × 9.97M) | all-pairs impossible; sub-linear blocking |
| Singletons | 5.6% of S1 entities; 89% have ≥2 matches, mean 3.46 | recall matters: predict **sets**, never only the best candidate |
| Both sources | 80% of matched entities match records in S2 **and** S3 | an entity's copies can support each other |
| One owner | no S2/S3 record belongs to two S1 entities (0 violations) | competing S1 claims are informative |
| Names repeat | 36% (US) / 44% (India) of S1 entities share their exact name with another S1 entity | a name match confirms, it does not identify |
| Addresses nearly unique | 5% of S1 entities share an address (max 14) | the address is the primary identity key |
| Cross-script names | 0% of S1 names, 20.5% (S2) / 11.6% (S3) of Indian names are in Devanagari, Tamil, Telugu, Kannada, ... | character n-grams share nothing across scripts |
| Address scripts | only 17 distinct Indic tokens occur in addresses, all state names | addresses stay Latin: the cross-script anchor |
| Missing address | 3.4% of S2/S3 records | name-only records must still be handled |
| France | 15% of test S1 (259k), absent from training | no country-specific features or rules learned from US/India |
| Record density | train 4.67 S2+S3 records per S1 entity; test 5.53-5.82 (every country) | ~2.3 unmatched decoys per test entity vs 1.2 in train: the test is harder than any random validation split |

Noise observed in matched pairs: typos, spurious accents, word reordering, legal-suffix flips (`LLC Orellana …`), appended generic words (`… Center`, `… Services`), bracket junk (`[[LLC]]`), literal `null` / `<NULL>`, phone numbers inside names, digit-for-letter substitutions (`0xford`, `6ulf`), padded or truncated house numbers (`0112`/`112`, `16076`/`1607`), and domain names used as names.

The unlabelled Source-2/3 records include **deliberate decoys**: records for a different business that differ from a true copy by one house number (`6850` vs `6855 Cherry Dr`) or one name word (`Solutions Trinity Energy` vs `Solutions Trinity Group`). 45% of the first model's false merges were records that belong to another Source-1 entity; the rest belong to no entity.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + F0.5-optimal set selection.
**Core Innovation:** (1) transliteration learned from training pairs; (2) blocking on name and address as one document; (3) reverse features from full-density blocking; (4) stage-2 group features over stage-1 probabilities; (5) decoy-branch augmentation that makes training and model selection match the test set's decoy density.

---

## 3. Candidate Generation (Blocking)

**Normalization** (`normalize.py`): Indic-script tokens are converted to Latin with a dictionary learned by aligning training pairs of (Latin S1 name, Indic S2/S3 name) position by position (1,347 entries; covers 100% of held-out validation tokens and 96.4% of test tokens; unseen words fall back to rule-based IAST transliteration). Then Unicode NFKD with accents stripped, junk removal, abbreviation canonicalization (street types, US and Indian states, French street types and legal forms), leading-zero removal in numbers, and digit-for-letter repair inside name words.

**Blocking keys used:** one TF-IDF index per (country, target source) over the word tokens of `normalized name + normalized address`; top 20 candidates per S1 entity per source, cosine similarity via sparse top-k multiplication. Terms in more than 200,000 documents (about 14 tokens such as `private`, `limited`, state codes) are dropped. Country only partitions the search; the code iterates over whatever country labels appear, so France is handled like any other country.

Why a combined document: a name-only index drowns in namesakes and an address-only index in co-located businesses. On 4,000 validation entities the true match ranked in the top 10 for 97.2% of pairs with the combined key, versus 86.9% (address only) and 66.4% (name words only).

- **Candidate pairs generated:** 40 per S1 entity (20 from S2, 20 from S3); about 69M for the test set.
- **How true matches were not lost:** recall was measured before any modelling. On 22,068 validation entities: 98.1% of true pairs are among the candidates (US 98.8%, India 97.1%); a perfect matcher on these candidates would score F0.5 = 0.9935.

---

## 4. Matching Model

**Features used** (`features.py`, `reverse.py`, `stack.py`):

- **Name:** fuzzy ratio, token-sort, token-set and partial ratio (full name); ratio, Jaro-Winkler, token-set and Jaccard (core name without legal/generic words); first-word match; exact match; share of distinctive S1 words missing from the candidate; count of unexplained extra words; Indic-script flag.
- **Address:** fuzzy ratio, token-set, partial ratio, Jaccard; word overlap ignoring numbers; empty-address flag; house numbers: share found exactly, conflict, truncation, numeric gap (adjacent buildings vs unrelated numbers), best fuzzy match; postcode match / conflict.
- **Rarity:** how many S1 entities share the core name / the address; how many S2/S3 records share the candidate's name; "unique name and exact name match" flag (94-99% match rate even without an address).
- **Blocking context:** similarity score, rank, gap to the entity's best candidate.
- **Group agreement:** how many of the entity's other candidates share this candidate's house-number set or core name, and whether it is the most common one.
- **Reverse features:** every S1 entity of the split is blocked, so for each candidate record we know how many S1 entities retrieved it, the best competing score, whether this S1 is its best match, and the margin.
- **Stage 2:** the stage-1 probability, its rank and gap within the entity, the sum of probabilities, probability-weighted support from candidates sharing numbers or name, and the best probability in the other source and among the other candidates of the same source.

**Model type:** LightGBM binary classifiers (learning rate 0.05, 255 leaves). Stage 1 uses pair features plus reverse features. Stage 2 adds the group summaries of stage-1 probabilities; on training entities these come from 3-fold out-of-fold models, so stage 2 is trained on the same kind of probabilities it sees at test time.

**Decoy-branch augmentation** (`augment.py`). Our first leaderboard submission scored 0.95 against 0.979 on validation. Measuring the test files explained it: they hold ~2.3 unmatched decoys per Source-1 entity against 1.2 in training, and the extra decoys arrive as *branches*: two or three records at a near-miss address that back each other up (for Source-1 `Rorie White Preferred Beverage, 10154 Rocky Point Rd`, the test holds `…Corp.`, `…uptown` and `…Midtown`, all at **10167**). The group features learned on training, where a decoy usually stands alone, voted for such branches. We reproduce the shift from training data alone: every pure decoy candidate (label 0, owned by no Source-1 entity) is cloned into the other source with identical pair features, and the group features are recomputed. Cloning half of them on the validation slice reproduces the leaderboard behaviour of the unaugmented model (3.463 matches kept per entity vs 3.456 on test; F0.5 0.9504 vs 0.95 on the leaderboard). The final model is trained with the same augmentation (rate 0.5), and the decision rule is selected on this test-like validation copy.

**Threshold selection method:** chosen on the test-like validation copy among a fixed threshold, two thresholds, plug-in expected-F0.5 set selection, and a *gated* variant. The gated rule won: an entity gets any match only if its best candidate clears a strict threshold (0.60), which protects singletons; the number of matches is then the prefix of probability-sorted candidates that maximizes expected F0.5.

---

## 5. Results & Error Analysis

Validation: 54,944 held-out S1 entities (entities, not pairs, are held out; blocking and all features are recomputed for them).

| Step | Validation F0.5 |
|---|---|
| Predict "no match" everywhere | 0.0557 |
| Stage 1, base features | 0.9706 |
| + decoy features (missing words, house-number conflict/truncation, rarity) | 0.9739 |
| + group agreement, gated rule | 0.9755 |
| + stage 2 | 0.9775 |
| + reverse features | 0.9792 |
| Blocking ceiling (perfect matcher) | 0.9935 |

Plain validation hides the test's decoy density, so the last step is measured on both copies:

| Model | Plain validation | Test-like validation | Leaderboard |
|---|---|---|---|
| Two-stage + reverse features | 0.9792 | 0.9504 | 0.95 |
| + decoy-branch augmentation (final) | 0.9786 | **0.9803** | (submitted) |

- **F_0.5 Score (macro):** 0.9786 on plain validation, 0.9803 on the test-like validation copy.
- **Common false positives (wrong merges):** sibling decoys: the same street with an adjacent or truncated house number, or a near-identical name differing by one word. Reverse features halved them (1,577 → 764 pairs).
- **Common false negatives (missed matches):** records without an address whose name is shared by several S1 entities. 48% of the remaining misses are of this kind; with no address and a namesake, nothing in the record identifies the owner, and under F0.5 guessing costs more than it gains.

**Measured but not shipped: a second blocking pass.** Two cheap extra keys, words cut to their first four letters (`specilists`→`spec`, `1500d`→`1500`) and the Source-1 name searched against records with no address, raise validation pair recall from 98.0% to 99.0% and the blocking ceiling from 0.9935 to 0.9969 (64 instead of 40 candidates per entity). The full test run needs ~7 hours on our laptop and did not fit before the deadline; the code is on the `union-blocking-wip` branch of our repository.

France cannot be scored without labels. As a check, on 2,100 test entities France received matches at the same rate as the training countries (5.3% predicted empty vs 4.4-5.4% for US and India; 3.30 matches per entity vs 3.22-3.29).

---

## 6. Conclusion

Measuring the data first shaped every decision: addresses rather than names identify businesses, transliteration can be learned from the training pairs themselves, and the hardest errors are decoys that only other Source-1 entities or the entity's other copies can expose. The largest single lesson came from the leaderboard: a random validation split looked 0.03 better than the test because the test carries twice the decoys, in branches. Measuring that shift and simulating it in training fixed the gap on our test-like validation (0.9504 → 0.9803). Most of the remaining loss is records without an address whose name is shared by several entities, and true matches the first blocking key misses, which a second key recovers.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the full pipeline (`src/`), `README.md` with step-by-step reproduction and timings, and `requirements.txt` with pinned versions. Entry points, in order: `convert_to_parquet.py`, `normalize.py`, `normalize_all.py`, `make_candidates.py dev`, `full_blocking.py train`, `train_matcher.py --rev`, `stack.py --rev --aug=0.5`, `full_blocking.py test`, `test_features.py`, `predict_test.py`.

### B. Compliance

- **No external data, APIs or lookups.** Every model, dictionary and statistic is derived from the provided training and test files. The decoy-branch augmentation only copies existing training candidates; the test files were used for unlabelled counts (records per entity), never for labels. The transliteration dictionary is learned from training pairs; the fallback transliteration is a rule-based character mapping (`indic-transliteration`, MIT; `anyascii`, ISC), not a lookup of any business.
- **Models:** LightGBM (MIT) gradient-boosted trees, far below 8B parameters. No pretrained language models are used.
- **Libraries:** polars, pyarrow, scikit-learn, scipy, numpy, rapidfuzz, sparse_dot_topn, lightgbm (MIT / BSD / Apache-2.0).
