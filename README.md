# Data Career Intelligence — Job Role Classification

Academic machine-learning team project investigating whether job postings can be classified into four data-career roles from their descriptions and skills.

## Task
Predict **Data Analyst**, **Data Engineer**, **Data Scientist**, or **Machine Learning Engineer** from job-posting data while reducing leakage from explicit role phrases in job descriptions.

## Verified core benchmark
The verified core reported in this repository compares skills-only baselines, masked and unmasked TF-IDF models, and a hybrid text-plus-skills model. The selected model is **masked word 1–2 gram TF-IDF + LinearSVC**.

| Metric | Reported test result |
|---|---:|
| Accuracy | 90.37% |
| Macro F1 | 86.29% |
| Number of test observations | 706 |

These are **reported results from the included project artifacts**, not results rerun for this GitHub package. The unmasked score is a leakage diagnostic; it is not the selected model.

See `docs/FINAL_RESULTS_SUMMARY.md` and the plots under `figures/final_core/`.

## Repository contents
- `src/` — feature extraction, preprocessing, model training, and evaluation modules from the archived project.
- `run_verified_core.py` — verified-core benchmark entry point.
- `run_all.py` — original orchestration script; **some stages reference files absent from the supplied ZIP and are not currently runnable**.
- `docs/` — methodology and verified evaluation notes.
- `results/final_core/` — aggregate metrics, not raw job descriptions.
- `figures/final_core/` — model comparisons, masking diagnostics and confusion matrix.
- `models/final_core/` — selected model and vectorizer artifacts, included for provenance. Do not load unknown `.joblib` files from untrusted sources.

## Environment and running

```bash
python -m pip install -r requirements.txt
python run_verified_core.py
```

**Important:** The benchmark runner requires the original **January 2024 raw job-posting data**, which is **not included** in this public package. The runner may fail with missing input data until the raw dataset is obtained and placed in the structure expected by the original code. See `src/config.py` and the methodology documents for data references. The archived `run_all.py --all` flow additionally references missing modules. This upload is therefore a **code-and-results portfolio snapshot**, not yet a one-command, end-to-end reproducible release.

## Methodology and limitations
- Explicit role terms are masked to reduce label leakage.
- Development data are divided into training, validation, and testing splits.
- Stronger performance for unmasked text is treated as a leakage warning, not an improvement.
- The uploaded ZIP does not contain every component of the group's broader final presentation, including independent external-dataset evaluations.
- No claims about real-world generalization are made beyond the verified core benchmark described in the included documents.

## Team attribution
Academic group work. Describe individual contribution separately and accurately on your CV; this snapshot does not establish personal ownership of every module.

## Next steps
Restore the missing source modules and dataset instructions, consolidate reproducible tests, verify saved models against the evaluation results, and add a working interactive demo.
