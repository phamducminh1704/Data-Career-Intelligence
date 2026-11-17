# Final Core Verification — Priority Pass 1

This document records the **verified, slide-safe core results** after taking over the final project. It intentionally separates evidence that has been rerun from legacy/placeholder artifacts.

## 1. What is verified now

The final classification task is:

> Predict one of four data-career roles — Data Analyst, Data Engineer, Data Scientist, or Machine Learning Engineer — from non-title job-posting information.

The label is derived conservatively from the job title, but **job title is never used as a model feature**. The core benchmark uses the original January 2024 dataset because it has adequate class counts for all four roles and allows a scientifically valid train/validation/test experiment.

## 2. Verified preprocessing checkpoints

The January 2024 data was rebuilt from the three raw CSV files:

- raw merged rows: **12,217**
- after exact normalized-description deduplication: **9,640**
- after requiring at least 200 description characters: **9,629**
- final four-role observations: **3,530**

Final class counts:

- Data Analyst: **1,490**
- Data Engineer: **1,116**
- Data Scientist: **698**
- Machine Learning Engineer: **226**

No exact normalized job-description duplicates remain after cleaning.

## 3. Leakage control

A major methodological issue is that many job descriptions repeat the job title. Before masking, the exact title appears inside the description for **46.5%** of retained postings.

Final leakage control:

1. remove the exact job title string from the text;
2. mask explicit role phrases such as `Data Scientist`, `Data Engineers`, `ML Engineer`, and `MLE`;
3. never feed `job_title` into X;
4. create strict 40-skill features from the **masked description**, not from title text.

Verification: after masking, exact title occurrences and the explicit role-name patterns checked by the pipeline are **0**.

## 4. Split design

A fixed stratified split with random seed 42 is used:

| Split | Data Analyst | Data Engineer | Data Scientist | ML Engineer | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,043 | 782 | 488 | 158 | 2,471 |
| Validation | 149 | 111 | 70 | 23 | 353 |
| Test | 298 | 223 | 140 | 45 | 706 |

The validation set is used to choose hyperparameters. The test set is evaluated after selection.

## 5. Verified model comparison

| Model | Accuracy | Macro F1 | Status |
|---|---:|---:|---|
| TF-IDF + LinearSVC, **unmasked** | 0.9561 | 0.9379 | Diagnostic only; title leakage present |
| **TF-IDF + LinearSVC, masked** | **0.9037** | **0.8629** | **Selected final core model** |
| Hybrid masked TF-IDF + strict 40 skills | 0.8909 | 0.8522 | Valid, but did not beat text-only |
| Legacy 40 skills + Logistic Regression | 0.7861 | 0.7456 | Midterm-style baseline |
| Strict masked-description 40 skills + Logistic Regression | 0.7890 | 0.7433 | Leakage-controlled skill baseline |
| Strict skills + Decision Tree | 0.7210 | 0.6661 | Nonlinear interpretable baseline |

The unmasked model is **not eligible** as the final model. Masking explicit title/role proxies reduces Macro F1 by about **7.50 percentage points** (0.9379 → 0.8629). That drop is evidence that title leakage materially inflated the easy version of the task.

## 6. Selected model: per-class results

Selected model: **masked TF-IDF word 1–2 grams + LinearSVC**, 5,000 maximum features, `class_weight='balanced'`, validation-selected `C=0.25`.

| Class | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| Data Analyst | 0.925 | 0.956 | 0.941 | 298 |
| Data Engineer | 0.940 | 0.915 | 0.927 | 223 |
| Data Scientist | 0.852 | 0.821 | 0.836 | 140 |
| ML Engineer | 0.739 | 0.756 | 0.747 | 45 |

Bootstrap 95% interval (1,000 test-set resamples):

- Accuracy: **0.904** [0.882, 0.925]
- Macro F1: **0.863** [0.828, 0.896]

The minority ML Engineer class remains the hardest class, which is why Macro F1 is retained as the primary metric.

## 7. Confusion/error pattern

Selected-model confusion matrix:

| Actual \ Predicted | Analyst | Engineer | Scientist | ML Engineer |
|---|---:|---:|---:|---:|
| Data Analyst | 285 | 5 | 8 | 0 |
| Data Engineer | 6 | 204 | 8 | 5 |
| Data Scientist | 15 | 3 | 115 | 7 |
| ML Engineer | 2 | 5 | 4 | 34 |

Largest error pair: **Data Scientist → Data Analyst (15 cases)**. DS ↔ MLE confusion is present but is not the single largest error pair in this test set. The earlier presentation draft overstated DS/MLE as the dominant confusion and should not be reused.

## 8. Important result about the proposed hybrid

The proposed hybrid model (masked TF-IDF + strict 40 skills) is scientifically valid but **does not outperform the masked text-only model**:

- Hybrid Macro F1: 0.8522
- Masked text-only Macro F1: 0.8629

This is a useful final-project finding, not a failure. It shows that the sparse text representation already captures most of the discriminative signal contained in the handcrafted 40-skill vector. The final presentation should report this honestly rather than claiming the hybrid is superior.

## 9. What is NOT verified yet

The following outputs from the inherited package are **not slide-safe** yet:

- `results/deprecated_placeholders/model_comparison_PLACEHOLDER_DO_NOT_USE.csv`
- `results/deprecated_placeholders/temporal_tests_PLACEHOLDER_DO_NOT_USE.csv`
- `results/deprecated_placeholders/external_tests_PLACEHOLDER_DO_NOT_USE.csv`
- old `docs/deprecated/PRESENTATION_PACKAGE_OLD_UNVERIFIED.md`

The previous `protocols.py` wrote hard-coded placeholder metrics. Those values must never be presented as empirical results.

## 10. Temporal-data limitation discovered

The Tinix source is large overall (606,878 rows), but conservative four-role title mapping yields very small historical samples for some target classes. Existing denominator artifacts show:

| Role | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|
| Data Analyst | 35 | 29 | 59 | 94 | 192 |
| Data Engineer | 4 | 7 | 14 | 59 | 87 |
| Data Scientist | 1 | 0 | 2 | 12 | 77 |
| ML Engineer | 0 | 1 | 0 | 2 | 12 |

Therefore a four-class **train 2022–2024 → test 2026** classifier is not scientifically defensible under the current conservative label rules: the historical train period contains only 3 Data Scientist and 1 ML Engineer observations.

Temporal skill-demand analysis can still be valuable after rerunning the corrected extractor and applying minimum-denominator rules, but temporal classification is currently demoted from a mandatory result to an optional/supplementary experiment.

## 11. Priority status

### Satisfied for the professor's requested final structure

- clear prediction problem;
- verified preprocessing counts for the main modeling dataset;
- multiple baselines;
- a selected model;
- validation/test protocol;
- accuracy, Macro F1, per-class metrics, confusion matrix;
- leakage analysis;
- error analysis;
- an honest model-comparison result.

### Still required before slides are frozen

1. finalize related/SOTA literature section;
2. decide whether to run a transformer comparator or present transformer literature as related SOTA only;
3. rerun Tinix trend extraction with `requirements` included and minimum sample thresholds, if raw/new-source data is made available in an environment with Parquet support;
4. run at least one real external/OOD evaluation on VietJobs or NextGig if adequate four-role labels are available;
5. update the final presentation package only from verified artifacts.

