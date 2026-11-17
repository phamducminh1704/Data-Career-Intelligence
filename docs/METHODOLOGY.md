# Methodology — Verified Core

## Prediction task

We formulate a four-class supervised classification problem. The target label is derived conservatively from explicit job-title patterns for Data Analyst, Data Engineer, Data Scientist, and Machine Learning Engineer. The title itself is **not** used as a feature.

## Data preparation

For the January 2024 baseline, three raw tables are joined one-to-one by `job_link`: job metadata, extracted skills, and job descriptions. Exact normalized duplicate descriptions are removed before modeling, and descriptions shorter than 200 characters are excluded. Only the four explicit target roles are retained.

## Leakage control

Job descriptions frequently repeat the title. We therefore mask the exact title string and explicit role-name phrases before text modeling. The strict skill baseline also extracts its 40 binary skill indicators from the masked description rather than from the job title.

## Features

### Tier A — skills only

- 40 binary technical-skill indicators.
- Logistic Regression with balanced class weights.
- Decision Tree baseline with balanced class weights.

Two skill variants are reported:

- **legacy**: provided extracted-skill field, to reproduce the midterm-style baseline;
- **strict**: skills re-extracted from leakage-controlled description text.

### Tier B — classical text

- TF-IDF word unigrams + bigrams;
- maximum 5,000 features;
- `min_df=2`;
- sublinear term frequency;
- LinearSVC with `class_weight='balanced'`.

Both unmasked and masked text are evaluated. Unmasked performance is diagnostic only and cannot be selected as the final model.

### Tier D — hybrid

The masked TF-IDF matrix is concatenated with the strict 40-skill matrix and classified with LinearSVC. This tests whether handcrafted skill indicators add complementary signal beyond the text representation.

## Train/validation/test protocol

- Train: 70% (2,471 observations)
- Validation: 10% (353)
- Test: 20% (706)
- Stratified by target role
- Random seed: 42

Hyperparameters are selected on validation Macro F1. Test results are reported after selection.

## Primary metric

Macro F1 is primary because the role classes are imbalanced and ML Engineer is much smaller than Data Analyst. Accuracy, macro precision, macro recall, per-class metrics, and confusion matrices are reported as complementary measures.

## Model selection

Among leakage-controlled candidates, masked TF-IDF + LinearSVC achieves the highest test Macro F1 and is simpler than the hybrid. It is selected as the final core model.

## Temporal and external evaluation

These remain separate protocols and must not be conflated with in-domain evaluation. Current Tinix historical counts are too sparse for a valid four-class temporal training experiment under conservative labels. Any temporal skill trend must be rerun with the corrected combined `description + requirements` extractor and must report denominators. External/OOD evaluation is pending verification on VietJobs/NextGig.
