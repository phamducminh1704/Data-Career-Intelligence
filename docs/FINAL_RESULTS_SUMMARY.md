# Final Results Summary — Verified Core Only

## Selected final core model

**Masked TF-IDF word 1–2 grams + LinearSVC**

- Accuracy: **0.9037**
- Macro Precision: **0.8641**
- Macro Recall: **0.8620**
- Macro F1: **0.8629**
- Test observations: **706**

Bootstrap 95% interval:

- Accuracy: [0.882, 0.925]
- Macro F1: [0.828, 0.896]

## Model ladder

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Unmasked TF-IDF + LinearSVC | 0.9561 | 0.9379 |
| **Masked TF-IDF + LinearSVC** | **0.9037** | **0.8629** |
| Hybrid masked TF-IDF + strict skills | 0.8909 | 0.8522 |
| Legacy skills LR | 0.7861 | 0.7456 |
| Strict masked-description skills LR | 0.7890 | 0.7433 |
| Strict skills Decision Tree | 0.7210 | 0.6661 |

The unmasked model is shown only as a leakage diagnostic and is not eligible for final selection.

## Main interpretation

Moving from 40 binary skills to leakage-controlled TF-IDF text substantially improves role classification. Adding the same handcrafted skill vector back to the text representation does not improve the selected model. The simplest strong, defensible model is therefore masked TF-IDF + LinearSVC.

## Artifacts

Verified numerical artifacts are under `results/final_core/`; figures are under `figures/final_core/`; selected model/vectorizer artifacts are under `models/final_core/`.
