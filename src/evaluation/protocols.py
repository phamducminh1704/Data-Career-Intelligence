"""Evaluation protocol aggregator.

This module must never fabricate metrics. It aggregates verified in-domain results and
records temporal/external protocols as pending until actual predictions exist.
"""
import os
import pandas as pd
from config import RESULTS_DIR


def run_protocols():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    verified = os.path.join(RESULTS_DIR, 'final_core', 'model_comparison_verified.csv')
    if not os.path.exists(verified):
        raise FileNotFoundError(
            'Verified core comparison is missing. Run the verified core benchmark before evaluation aggregation.'
        )

    comp = pd.read_csv(verified)
    comp.insert(1, 'test_type', 'in_domain_verified')
    comp.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

    pd.DataFrame([{
        'status': 'NOT_RUN',
        'protocol': 'temporal_holdout',
        'reason': 'Current conservative Tinix historical labels are too sparse for a valid 4-class 2022-2024 training experiment. Rerun only after a defensible protocol is established.'
    }]).to_csv(os.path.join(RESULTS_DIR, 'temporal_tests.csv'), index=False)

    pd.DataFrame([{
        'status': 'NOT_RUN',
        'protocol': 'external_ood',
        'reason': 'External/OOD predictions on VietJobs/NextGig have not yet been verified on a consistent four-role label set.'
    }]).to_csv(os.path.join(RESULTS_DIR, 'external_tests.csv'), index=False)

    print('Evaluation aggregation complete. No placeholder metrics were generated.')


if __name__ == '__main__':
    run_protocols()
