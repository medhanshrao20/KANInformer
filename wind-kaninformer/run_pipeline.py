"""
Master pipeline script.
Runs all 8 preprocessing steps then trains and evaluates KANInformer.

Usage:
  python run_pipeline.py [--force] [--season spring]

--force  : Pass --force flag to every step (recompute from scratch)
--season : Run only for a single season
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true',
                   help='Force recomputation of all steps')
    p.add_argument('--season', type=str, default=None,
                   help='Run only for a specific season')
    return p.parse_args()


def run_step(label, script, extra_args=None):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f'\n  [{label}] completed in {elapsed:.1f}s')
    return result


def check_data_file():
    if not os.path.exists(config.RAW_CSV):
        print(f'\nERROR: Data file not found at {config.RAW_CSV}')
        print('Please download CIMIS hourly data for Station 47 (Brentwood)')
        print('Period: December 1 2020 to December 1 2021')
        print('URL: https://cimis.water.ca.gov/')
        print('Place the downloaded file as: data/hourly.csv')
        sys.exit(1)
    print(f'Data file found: {config.RAW_CSV}')


def main():
    args = parse_args()

    check_data_file()

    extra = []
    if args.force:
        extra.append('--force')
    if args.season:
        extra.extend(['--season', args.season])

    steps = [
        ('STEP 1: Season Split',          'step1_prepare_seasons.py'),
        ('STEP 2: PCC Selection',         'step2_pcc_selection.py'),
        ('STEP 3: VMD Decomposition',     'step3_vmd_decompose.py'),
        ('STEP 4: Sample Entropy',        'step4_sample_entropy.py'),
        ('STEP 5: Component Aggregation', 'step5_component_aggregation.py'),
        ('STEP 6: EWT Decomposition',     'step6_ewt_decompose.py'),
        ('STEP 7: Build Feature Matrix',  'step7_build_feature_matrix.py'),
        ('STEP 8: Normalize + Windows',   'step8_normalize_and_window.py'),
        ('TRAINING + EVALUATION',         'KANInformer.py'),
    ]

    # Steps 1-2 don't accept --season in the same way; always pass --force if needed
    # but not --season (they process all seasons at once for consistency)
    season_capable = {
        'step3_vmd_decompose.py', 'step4_sample_entropy.py',
        'step5_component_aggregation.py', 'step6_ewt_decompose.py',
        'step7_build_feature_matrix.py', 'step8_normalize_and_window.py',
        'KANInformer.py',
    }

    total_start = time.time()

    for label, script in steps:
        step_extra = list(extra)
        if args.season and script not in season_capable:
            # Steps 1 and 2 process all seasons together
            step_extra = ['--force'] if args.force else []
        run_step(label, script, step_extra if step_extra else None)

    total_elapsed = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'  PIPELINE COMPLETE')
    print(f'  Total time: {total_elapsed/60:.1f} minutes')
    print(f'{"="*60}')

    # Print final results if available
    import pandas as pd
    results_path = os.path.join(config.RESULTS_DIR, 'table9_results.csv')
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        print('\nFinal Results (Table 9 comparison):')
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
