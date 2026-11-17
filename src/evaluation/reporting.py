"""Safe reporting helper.

The inherited version overwrote completed documentation with placeholder text. This
replacement never overwrites scientific documents. It writes only a small status file
pointing to the verified evidence package.
"""
import os
from datetime import datetime, timezone
from config import BASE_DIR


def run_reporting():
    docs_dir = os.path.join(BASE_DIR, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, 'REPORTING_STATUS.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Reporting Status\n\n')
        f.write(f'Updated: {datetime.now(timezone.utc).isoformat()}\n\n')
        f.write('Authoritative verified core documents:\n\n')
        f.write('- `FINAL_CORE_VERIFICATION.md`\n')
        f.write('- `FINAL_RESULTS_SUMMARY.md`\n')
        f.write('- `METHODOLOGY.md`\n')
        f.write('- `LESSONS_LEARNED.md`\n')
        f.write('- `SOTA_RELATED_WORK.md`\n')
        f.write('- `PRESENTATION_PACKAGE_VERIFIED_CORE.md`\n\n')
        f.write('Do not use files under `docs/deprecated/` or `results/deprecated_placeholders/`.\n')
    print('Safe reporting status generated:', path)


if __name__ == '__main__':
    run_reporting()
