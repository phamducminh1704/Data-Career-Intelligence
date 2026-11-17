# Lessons Learned — Verified

1. **Leakage control matters more than headline accuracy.** Nearly half of retained descriptions repeat the exact title. The unmasked text model reaches Macro F1 0.938, but masking direct role/title proxies lowers it to 0.863. The lower number is the scientifically defensible one.

2. **A stronger model does not have to be more complicated.** The masked TF-IDF + LinearSVC model outperforms the proposed hybrid. Adding the 40 handcrafted skill features did not improve the result.

3. **Macro F1 is necessary.** Accuracy alone hides weaker performance on the minority ML Engineer class.

4. **Data volume is not the same as usable labeled volume.** Tinix contains more than 600k postings, yet conservative mapping yields very few historical Data Scientist and ML Engineer examples. A huge raw dataset can still be unsuitable for a specific supervised-learning protocol.

5. **Temporal and source shift must be separated.** A difference between 2024 Kaggle/LinkedIn-derived data and 2025/2026 Vietnamese or ATS data may reflect time, region, source, language, or collection method. Cross-source differences cannot automatically be called market trends.

6. **Requirements text is essential for skill extraction.** The inherited pipeline ignored Tinix `requirements`, which is exactly where many technical requirements are stated. The extractor has been corrected to use leakage-controlled combined description + requirements text.

7. **Negative experimental results are useful.** The hybrid model not beating text-only is a legitimate finding and should be reported rather than hidden.
