"""
STEP 7 — Build final feature matrix.

Combines selected meteo variables + low-SE IMFs + EWT modes + WS (last column).

Run independently:  python step7_build_feature_matrix.py [--force]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    p.add_argument('--season', type=str, default=None)
    return p.parse_args()


def check_outputs_exist(season):
    path = os.path.join(config.SEASONS_DIR, season, 'feature_matrix.csv')
    return os.path.exists(path)


def load_season_features(season):
    """Load SEASON_FEATURES from config (may have been updated by step2)."""
    # Re-import to get updated values written by step2
    import importlib
    cfg = importlib.import_module('config')
    importlib.reload(cfg)
    return getattr(cfg, 'SEASON_FEATURES', {}).get(season, config.EXPECTED_FEATURES.get(season, []))


def process_season(season, args):
    print(f'\n=== {season.upper()} ===')

    if not args.force and check_outputs_exist(season):
        print(f'  Feature matrix already exists for {season}. Skipping.')
        return

    # Load raw seasonal data (all rows, no NaN drop yet)
    raw_path = os.path.join(config.SEASONS_DIR, season, 'raw.csv')
    if not os.path.exists(raw_path):
        print(f'  ERROR: {raw_path} not found. Run step1 first.')
        return
    season_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    N_raw = len(season_df)
    print(f'  Raw season rows: {N_raw}')

    # Load decomposition arrays
    decomp_dir = os.path.join(config.DECOMP_DIR, season)
    low_se_imfs = np.load(os.path.join(decomp_dir, 'low_se_imfs.npy'))   # (num_low, N_vmd)
    ewt_modes = np.load(os.path.join(decomp_dir, 'ewt_modes.npy'))        # (8, N_ewt)

    N_vmd = low_se_imfs.shape[1] if low_se_imfs.ndim == 2 else 0
    N_ewt = ewt_modes.shape[1]

    print(f'  low_se_imfs shape: {low_se_imfs.shape}')
    print(f'  ewt_modes shape:   {ewt_modes.shape}')

    # All arrays must align to same length — use minimum
    lengths = [N_raw, N_ewt]
    if low_se_imfs.shape[0] > 0:
        lengths.append(N_vmd)
    N = min(lengths)

    if N < N_raw:
        print(f'  NOTE: Trimming all arrays to N={N} (min of {lengths})')

    # Get selected meteo variables for this season
    selected_vars = load_season_features(season)
    print(f'  Selected meteo vars: {selected_vars}')

    # Build column dictionary in prescribed order:
    # [meteo vars] + [low-SE IMFs] + [EWT 1..8] + [WS]
    cols = {}

    for var in selected_vars:
        if var in season_df.columns:
            vals = season_df[var].values[:N]
            cols[var] = vals
        else:
            print(f'  WARNING: {var} not found in raw.csv, filling with zeros')
            cols[var] = np.zeros(N)

    num_low = low_se_imfs.shape[0]
    for i in range(num_low):
        cols[f'IMF_low_{i+1}'] = low_se_imfs[i, :N]

    for i in range(config.N_EWT):
        cols[f'EWT_{i+1}'] = ewt_modes[i, :N]

    # WS is always last
    cols[config.TARGET_VAR] = season_df[config.TARGET_VAR].values[:N]

    feature_df = pd.DataFrame(cols, index=season_df.index[:N])

    # Drop rows with any NaN
    before_drop = len(feature_df)
    feature_df = feature_df.dropna()
    after_drop = len(feature_df)
    if before_drop != after_drop:
        print(f'  Dropped {before_drop - after_drop} NaN rows '
              f'({before_drop} -> {after_drop})')

    print(f'  Final feature matrix shape: {feature_df.shape}')
    print(f'  N_heng (feature count): {feature_df.shape[1]}')
    print(f'  Columns ({len(feature_df.columns)}): {list(feature_df.columns)}')

    # Save
    out_path = os.path.join(config.SEASONS_DIR, season, 'feature_matrix.csv')
    feature_df.to_csv(out_path)
    print(f'  Saved feature_matrix.csv to {out_path}')


def main():
    args = parse_args()
    seasons = [args.season] if args.season else config.SEASON_ORDER

    for season in seasons:
        if season not in config.SEASON_DATES:
            continue
        process_season(season, args)

    print('\nStep 7 complete.')


if __name__ == '__main__':
    main()
