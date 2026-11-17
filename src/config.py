import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Raw new-source data intentionally lives outside the project workspace.
# Prefer an explicit environment variable for portability:
#   Windows PowerShell: $env:FINAL_DATA_DIR='C:\path\to\final_data'
#   macOS/Linux: export FINAL_DATA_DIR=/path/to/final_data
_env = os.environ.get('FINAL_DATA_DIR')
_candidates = [
    Path(_env).expanduser() if _env else None,
    BASE_DIR.parent / 'final_data',
    BASE_DIR.parent / 'FINAL_DATA',
    BASE_DIR / 'final_data',
]
FINAL_DATA_DIR = next((p.resolve() for p in _candidates if p is not None and p.exists()),
                      (Path(_env).expanduser().resolve() if _env else (BASE_DIR.parent / 'final_data').resolve()))

MIDTERM_RAW = FINAL_DATA_DIR / '00_MIDTERM_JAN2024' / 'raw' / 'raw.zip'
TINIX_RAW = FINAL_DATA_DIR / '01_TINIX_VIETNAM_2022_2026' / 'raw' / 'data.parquet'
VIETJOBS_RAW = FINAL_DATA_DIR / '02_VIETJOBS_2025' / 'raw' / 'VietJobs.csv'
NEXTGIG_RAW = FINAL_DATA_DIR / '03_NEXTGIG_GLOBAL_JUNE2026' / 'raw' / 'nextgig_jobs_2026-06.parquet'

DATA_PROCESSED_DIR = str(BASE_DIR / 'data' / 'processed_by_source')
DATA_SPLITS_DIR = str(BASE_DIR / 'data' / 'splits')
DATA_AUDITS_DIR = str(BASE_DIR / 'data' / 'audits')
MODELS_DIR = str(BASE_DIR / 'models')
RESULTS_DIR = str(BASE_DIR / 'results')
FIGURES_DIR = str(BASE_DIR / 'figures')

# Keep string aliases for older modules using os.path APIs.
BASE_DIR = str(BASE_DIR)
MIDTERM_RAW = str(MIDTERM_RAW)
TINIX_RAW = str(TINIX_RAW)
VIETJOBS_RAW = str(VIETJOBS_RAW)
NEXTGIG_RAW = str(NEXTGIG_RAW)

RANDOM_SEED = 42
TARGET_ROLES = ['Data Analyst', 'Data Scientist', 'Data Engineer', 'Machine Learning Engineer', 'Other']
MIN_DESCRIPTION_LENGTH = 200
SPLIT_TEST_SIZE = 0.2
