"""
STEP 5 — Component Aggregation (CA).

Novel step from the paper: aggregate ALL high-SE IMFs into one fused signal,
keep low-SE IMFs as separate components.

Run independently:  python step5_component_aggregation.py [--force]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    p.add_argument('--season', type=str, default=None)
    return p.parse_args()


def check_outputs_exist(season):
    d = os.path.join(config.DECOMP_DIR, season)
    return (os.path.exists(os.path.join(d, 'fused_imf.npy')) and
            os.path.exists(os.path.join(d, 'low_se_imfs.npy')))


def process_season(season, args):
    print(f'\n=== {season.upper()} ===')

    if not args.force and check_outputs_exist(season):
        print(f'  CA outputs already exist for {season}. Skipping.')
        return

    decomp_dir = os.path.join(config.DECOMP_DIR, season)

    imfs_path = os.path.join(decomp_dir, 'imfs.npy')
    se_path = os.path.join(decomp_dir, 'se_values.npy')

    if not os.path.exists(imfs_path):
        print(f'  ERROR: {imfs_path} not found. Run step3 first.')
        return
    if not os.path.exists(se_path):
        print(f'  ERROR: {se_path} not found. Run step4 first.')
        return

    imfs = np.load(imfs_path)       # (K, N)
    se_values = np.load(se_path)    # (K,)
    K, N = imfs.shape

    print(f'  IMFs shape: {imfs.shape}')
    print(f'  SE values:  {[round(v, 4) for v in se_values]}')

    high_se_indices = [i for i, se in enumerate(se_values) if se > config.SE_THRESHOLD]
    low_se_indices = [i for i, se in enumerate(se_values) if se <= config.SE_THRESHOLD]

    print(f'  High-SE IMF indices (0-based): {high_se_indices}  '
          f'-> IMF{[i+1 for i in high_se_indices]}')
    print(f'  Low-SE  IMF indices (0-based): {low_se_indices}  '
          f'-> IMF{[i+1 for i in low_se_indices]}')

    # Edge case: no high-SE IMFs
    if len(high_se_indices) == 0:
        best_idx = int(np.argmax(se_values))
        print(f'  WARNING: No IMFs exceed SE threshold {config.SE_THRESHOLD}. '
              f'Using IMF{best_idx+1} (highest SE={se_values[best_idx]:.4f}) as fused_imf.')
        high_se_indices = [best_idx]
        low_se_indices = [i for i in range(K) if i != best_idx]

    # Aggregate high-SE IMFs
    fused_imf = np.sum(imfs[high_se_indices, :], axis=0)  # (N,)
    print(f'  fused_imf shape: {fused_imf.shape}  '
          f'(sum of {len(high_se_indices)} high-SE IMFs)')

    # Low-SE IMFs
    if len(low_se_indices) > 0:
        low_se_imfs = imfs[low_se_indices, :]  # (num_low, N)
    else:
        print('  WARNING: No low-SE IMFs. Creating empty placeholder.')
        low_se_imfs = np.zeros((0, N))
    print(f'  low_se_imfs shape: {low_se_imfs.shape}')

    np.save(os.path.join(decomp_dir, 'fused_imf.npy'), fused_imf)
    np.save(os.path.join(decomp_dir, 'low_se_imfs.npy'), low_se_imfs)
    print(f'  Saved fused_imf.npy and low_se_imfs.npy to {decomp_dir}')


def main():
    args = parse_args()
    seasons = [args.season] if args.season else config.SEASON_ORDER

    for season in seasons:
        if season not in config.SEASON_DATES:
            continue
        process_season(season, args)

    print('\nStep 5 complete.')


if __name__ == '__main__':
    main()
