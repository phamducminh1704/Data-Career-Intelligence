import re
import pandas as pd

# Role names and common grammatical variants that can directly reveal the title-derived label.
ROLE_PHRASES = [
    r'\bdata\s+scientists?\b',
    r'\bdata\s+engineers?\b',
    r'\bdata\s+analysts?\b',
    r'\bmachine\s+learning\s+engineers?\b',
    r'\bml\s+engineers?\b',
    r'\bmle\b',
    r'\bnhà khoa học dữ liệu\b',
    r'\bkỹ sư dữ liệu\b',
    r'\b(chuyên viên|nhân viên) phân tích dữ liệu\b',
    r'\bkỹ sư (machine learning|học máy)\b',
]


def mask_role_names(text, exact_title=None):
    if not isinstance(text, str):
        return '' if pd.isna(text) else str(text)
    out = text
    # Job descriptions often repeat the exact title. Remove it first because it is the strongest proxy.
    if isinstance(exact_title, str) and exact_title.strip():
        out = re.sub(re.escape(exact_title.strip()), ' [ROLE_TITLE] ', out, flags=re.IGNORECASE)
    for phrase in ROLE_PHRASES:
        out = re.sub(phrase, ' [ROLE] ', out, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', out).strip()


def apply_masking(df):
    """Create leakage-controlled text fields without destroying source text.

    The final classifier and strict skill extractor should use `text_combined_masked`.
    Original text remains available only for diagnostics/audit.
    """
    out = df.copy()
    titles = out.get('job_title_raw', pd.Series('', index=out.index)).fillna('').astype(str)
    desc = out.get('text_original_or_best_available', pd.Series('', index=out.index)).fillna('').astype(str)
    req = out.get('requirements_text', pd.Series('', index=out.index)).fillna('').astype(str)

    out['text_masked'] = [mask_role_names(t, title) for t, title in zip(desc, titles)]
    out['requirements_masked'] = [mask_role_names(t, title) for t, title in zip(req, titles)]
    out['text_combined_unmasked'] = (desc + ' ' + req).str.replace(r'\s+', ' ', regex=True).str.strip()
    out['text_combined_masked'] = (out['text_masked'] + ' ' + out['requirements_masked']).str.replace(r'\s+', ' ', regex=True).str.strip()
    return out
