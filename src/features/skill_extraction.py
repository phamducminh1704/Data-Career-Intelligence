import re
import pandas as pd
from features.skill_taxonomy import LEGACY_40_SKILLS


def build_regex_pattern(aliases):
    """Compile lexical-boundary-aware aliases.

    Prevents false positives such as ML in HTML/XML, SQL in NoSQL/MySQL,
    Java in JavaScript, or Git in GitHub/digital.
    """
    patterns = []
    for alias in aliases:
        escaped = re.escape(alias)
        patterns.append(r'(?<![a-z0-9+#])' + escaped + r'(?![a-z0-9+#])')
    return re.compile('|'.join(patterns), re.IGNORECASE)


def extract_skills(text, compiled_patterns):
    text = '' if pd.isna(text) else str(text)
    return {skill: int(pattern.search(text) is not None) for skill, pattern in compiled_patterns.items()}


def run_skill_extraction(df, *, text_col=None, include_provided_skills=False):
    """Create the 40 binary skill features.

    Strict final default: use the leakage-controlled combined description + requirements field.
    `provided_skills_raw` is excluded by default because its upstream extraction provenance may be
    unknown and can contain title-derived proxy information. Set include_provided_skills=True only
    for explicit legacy-reproduction experiments.
    """
    compiled_patterns = {skill: build_regex_pattern(aliases) for skill, aliases in LEGACY_40_SKILLS.items()}

    if text_col is None:
        if 'text_combined_masked' in df.columns:
            text_col = 'text_combined_masked'
        elif 'text_masked' in df.columns:
            text_col = 'text_masked'
        else:
            text_col = 'text_original_or_best_available'

    def get_text(row):
        t = '' if pd.isna(row.get(text_col)) else str(row.get(text_col))
        if include_provided_skills and pd.notna(row.get('provided_skills_raw')):
            t += ' ' + str(row.get('provided_skills_raw'))
        return t

    texts = df.apply(get_text, axis=1)
    features = texts.apply(lambda x: pd.Series(extract_skills(x, compiled_patterns)))
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
