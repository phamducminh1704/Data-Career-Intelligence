# Related / SOTA Work

This section is for methodological context. Reported scores from other papers are **not directly comparable** with our 4-class test score because datasets, labels, languages, and collection protocols differ.

## 1. JobBERT: Understanding Job Titles through Skills (2021)

**Decorte, Van Hautte, Demeester, and Develder.**

- Task: job-title representation / job-title normalization.
- Method: pretrained language-model representation enhanced with skill co-occurrence information from vacancies.
- Relevance: demonstrates that domain-specific job information and skill signals can improve generic text representations.
- Difference from our task: JobBERT focuses on title normalization, whereas our target is four-role classification from non-title posting information.
- Source: arXiv:2109.09605 — https://arxiv.org/abs/2109.09605

## 2. Classification of Job Offers into Job Positions Using O*NET and BERT Language Models (2025)

- Task: multiclass job-offer classification into O*NET occupational families.
- Models compared: BERT, RoBERTa, DeBERTa.
- Dataset: approximately 692,600 English job offers, organized into broad O*NET occupational families.
- Best reported model: DeBERTa, Macro F1 ≈ **0.767** on that study's task.
- Relevance: a recent transformer-based reference for occupational classification from job-offer text.
- Important fairness caveat: their 22-family O*NET task is much broader and different from our four data-career classes, so 0.767 must not be compared numerically as if it were the same benchmark.
- DOI: 10.32604/cmc.2025.070813
- Article: https://www.techscience.com/cmc/v86n2/64766/html

## 3. Multilingual hierarchical classification of job advertisements for job vacancy statistics (2024)

**Beręsewicz, Wydmuch, Cherniaiev, and Pater.**

- Task: multilingual hierarchical occupation coding of online job advertisements using transformer architectures.
- Contribution: shows that hierarchical occupation structure and multilingual modeling can improve job-ad classification and supports cross-country labor-market statistics.
- Relevance: directly motivates future regional/multilingual extensions of our project, especially Vietnam.
- Source: arXiv:2411.03779 — https://arxiv.org/abs/2411.03779

## How to compare fairly in the final presentation

Use two layers:

1. **Direct empirical comparison on our dataset:** Legacy skills LR vs strict skills LR vs Decision Tree vs masked TF-IDF LinearSVC vs hybrid.
2. **SOTA literature context:** transformer models such as BERT/RoBERTa/DeBERTa are established strong methods for occupational/job-offer classification. Their published scores are contextual references, not a leaderboard against our score.

A transformer comparator on our exact split would make the comparison stronger if time/compute permits, but the final report must not claim to beat SOTA merely because our four-class Macro F1 is numerically higher than a paper using 22 occupational families.
