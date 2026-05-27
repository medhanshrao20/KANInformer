"""
STEP 4 — Compute Sample Entropy for each VMD IMF.

Classifies IMFs as High (SE > SE_THRESHOLD) or Low unpredictability.

Run independently:  python step4_sample_entropy.py [--force]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    p.add_argument('--season', type=str, default=None)
    return p.parse_args()


def check_outputs_exist(season):
    return os.path.exists(os.path.join(config.DECOMP_DIR, season, 'se_values.npy'))


def compute_sample_entropy(signal, m=2, metric='chebyshev'):
    """Compute sample entropy using antropy library."""
    import antropy
    try:
        se = antropy.sample_entropy(signal, order=m, metric=metric)
        if np.isnan(se) or np.isinf(se):
            print(f'      WARNING: SE={se}, assigning 0.0')
            return 0.0
        return float(se)
    except Exception as e:
        print(f'      WARNING: Sample entropy failed: {e}. Assigning 0.0')
        return 0.0


def process_season(season, args):
    print(f'\n=== {season.upper()} ===')

    if not args.force and check_outputs_exist(season):
        print(f'  SE values already exist for {season}. Skipping.')
        return []

    imfs_path = os.path.join(config.DECOMP_DIR, season, 'imfs.npy')
    if not os.path.exists(imfs_path):
        print(f'  ERROR: {imfs_path} not found. Run step3 first.')
        return []

    imfs = np.load(imfs_path)
    K = imfs.shape[0]
    print(f'  IMFs shape: {imfs.shape}  (K={K})')

    se_values = np.zeros(K)
    rows = []

    for i in range(K):
        imf = imfs[i]
        print(f'    IMF{i+1}: computing SE...', end='', flush=True)
        se = compute_sample_entropy(imf)
        se_values[i] = se
        classification = 'High' if se > config.SE_THRESHOLD else 'Low'
        print(f'  SE={se:.4f}  [{classification}]')
        rows.append({
            'Season': season,
            'IMF_index': i + 1,
            'SE_value': round(se, 4),
            'Classification': classification,
        })

    # Log comparison with paper
    high_se = [i + 1 for i, se in enumerate(se_values) if se > config.SE_THRESHOLD]
    low_se = [i + 1 for i, se in enumerate(se_values) if se <= config.SE_THRESHOLD]
    print(f'  High-SE IMFs (SE > {config.SE_THRESHOLD}): {high_se}')
    print(f'  Low-SE IMFs:  {low_se}')

    out_dir = os.path.join(config.DECOMP_DIR, season)
    np.save(os.path.join(out_dir, 'se_values.npy'), se_values)
    print(f'  Saved se_values.npy shape {se_values.shape}')

    return rows


def main():
    args = parse_args()

    # Check if se_values_all_seasons.csv exists
    csv_path = os.path.join(config.RESULTS_DIR, 'se_values_all_seasons.csv')
    if not args.force and os.path.exists(csv_path):
        # Check individual season files too
        all_exist = all(check_outputs_exist(s) for s in config.SEASON_ORDER)
        if all_exist:
            print('Step 4 outputs already exist. Use --force to recompute.')
            return

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    seasons = [args.season] if args.season else config.SEASON_ORDER
    all_rows = []

    for season in seasons:
        if season not in config.SEASON_DATES:
            continue
        rows = process_season(season, args)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f'\nSaved SE results to {csv_path}')

    print('\nStep 4 complete.')


if __name__ == '__main__':
    main()
