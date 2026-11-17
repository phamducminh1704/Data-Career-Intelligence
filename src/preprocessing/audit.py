import os
import pandas as pd
import json
import re
from config import *

def conservative_role_label(title):
    if not isinstance(title, str):
        return 'Other'
    title = title.lower()
    
    # Priority order from midterm Appendix B + explicit Vietnamese equivalents
    if re.search(r'\b(machine learning|ml)\s*engineer\b', title) or re.search(r'\bkỹ sư (machine learning|học máy)\b', title):
        return 'Machine Learning Engineer'
    if re.search(r'\bdata scientist\b', title) or re.search(r'\bnhà khoa học dữ liệu\b', title):
        return 'Data Scientist'
    if re.search(r'\bdata engineer\b', title) or re.search(r'\bkỹ sư dữ liệu\b', title):
        return 'Data Engineer'
    if re.search(r'\bdata analyst\b', title) or re.search(r'\b(chuyên viên|nhân viên) phân tích dữ liệu\b', title):
        return 'Data Analyst'
    
    return 'Other'

def audit_source(source_name, file_path, file_type):
    print(f"Auditing {source_name}...")
    audit = {
        'source_name': source_name,
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
    }
    
    if file_type == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_type == 'csv':
        df = pd.read_csv(file_path, low_memory=False)
    else:
        raise ValueError("Unsupported file type")
        
    audit['row_count'] = len(df)
    audit['columns'] = list(df.columns)
    audit['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missingness
    missing = df.isnull().sum().to_dict()
    audit['missingness'] = missing
    
    # Exact Duplicates
    audit['exact_duplicate_rows'] = int(df.duplicated().sum())
    
    # We will identify specific columns based on source
    title_col = None
    desc_col = None
    year_col = None
    
    if source_name == 'TINIX':
        title_col = 'job_title'
        desc_col = 'job_description'
        year_col = 'year'
    elif source_name == 'NEXTGIG':
        title_col = 'title'
        desc_col = 'job_description'
        year_col = 'date_posted' # Will need to extract year
    elif source_name == 'VIETJOBS':
        title_col = 'job_title'
        desc_col = 'description'
    
    if title_col and title_col in df.columns:
        audit['duplicate_titles'] = int(df[title_col].duplicated().sum())
        df['target_role'] = df[title_col].apply(conservative_role_label)
        role_counts = df['target_role'].value_counts().to_dict()
        audit['preliminary_role_counts'] = role_counts
        
    if desc_col and desc_col in df.columns:
        audit['duplicate_descriptions'] = int(df[desc_col].astype(str).duplicated().sum())
        
    if year_col and year_col in df.columns:
        if source_name == 'NEXTGIG':
            years = pd.to_datetime(df[year_col], errors='coerce').dt.year
        else:
            years = df[year_col]
        audit['year_distribution'] = years.value_counts().to_dict()
        
    return audit

def main():
    os.makedirs(DATA_AUDITS_DIR, exist_ok=True)
    
    sources = [
        ('TINIX', TINIX_RAW, 'parquet'),
        ('VIETJOBS', VIETJOBS_RAW, 'csv'),
        ('NEXTGIG', NEXTGIG_RAW, 'parquet')
    ]
    
    all_audits = []
    for name, path, ftype in sources:
        try:
            audit = audit_source(name, path, ftype)
            all_audits.append(audit)
        except Exception as e:
            print(f"Failed to audit {name}: {e}")
            
    with open(os.path.join(DATA_AUDITS_DIR, 'audit_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(all_audits, f, indent=4, ensure_ascii=False)
        
    # Generate Markdown
    md_lines = ["# Data Audit Report", ""]
    for audit in all_audits:
        md_lines.append(f"## Source: {audit['source_name']}")
        md_lines.append(f"- **File:** {audit['file_name']} ({audit['file_size_mb']:.2f} MB)")
        md_lines.append(f"- **Rows:** {audit['row_count']}")
        md_lines.append(f"- **Columns:** {len(audit['columns'])}")
        md_lines.append(f"- **Exact Duplicates:** {audit.get('exact_duplicate_rows', 'N/A')}")
        
        md_lines.append("\n### Missingness (Top 10)")
        sorted_missing = sorted(audit['missingness'].items(), key=lambda x: x[1], reverse=True)[:10]
        for col, count in sorted_missing:
            md_lines.append(f"- `{col}`: {count} missing ({count/audit['row_count']*100:.2f}%)")
            
        if 'preliminary_role_counts' in audit:
            md_lines.append("\n### Preliminary Roles")
            for role, count in audit['preliminary_role_counts'].items():
                md_lines.append(f"- {role}: {count}")
                
        if 'year_distribution' in audit:
            md_lines.append("\n### Year Distribution")
            for year, count in audit['year_distribution'].items():
                md_lines.append(f"- {year}: {count}")
                
        md_lines.append("\n---\n")
        
    with open(os.path.join(BASE_DIR, 'docs', 'DATA_AUDIT.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
        
    print("Audit complete.")

if __name__ == '__main__':
    main()
