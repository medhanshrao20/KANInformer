"""
STEP 2 — Pearson Correlation Coefficient feature selection.

For each season, compute PCC between WS and the 9 other meteo variables.
Select features where abs(r) > PCC_THRESHOLD and p-value < PCC_PVALUE.
Split selected feature data 8:1:1 and save train/val/test CSVs.

Run independently:  python step2_pcc_selection.py [--force]
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    return p.parse_args()


def check_outputs_exist():
    for season in config.SEASON_ORDER:
        for f in ['train.csv', 'val.csv', 'test.csv']:
            if not os.path.exists(os.path.join(config.SEASONS_DIR, season, f)):
                return False
    return os.path.exists(os.path.join(config.RESULTS_DIR, 'pcc_results_all_seasons.csv'))


def compute_pcc(df, target='WS', candidates=None):
    """Compute PCC between target and each candidate variable."""
    if candidates is None:
        candidates = config.ALL_METEO_VARS
    results = []
    for var in candidates:
        if var not in df.columns:
            continue
        valid = df[[target, var]].dropna()
        if len(valid) < 10:
            print(f'    {var}: too few valid rows ({len(valid)}), skipping')
            results.append({'variable': var, 'r': np.nan, 'p': np.nan, 'selected': False})
            continue
        r, p = pearsonr(valid[target].values, valid[var].values)
        selected = (abs(r) > config.PCC_THRESHOLD) and (p < config.PCC_PVALUE)
        results.append({'variable': var, 'r': r, 'p': p, 'selected': selected})
        print(f'    {var}: r={r:+.4f}, p={p:.4e}, selected={selected}')
    return results


def split_dataframe(df):
    """Chronological 8:1:1 split. No shuffling."""
    n = len(df)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def update_config_season_features(season_features_dict):
    """Write SEASON_FEATURES back into config.py."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    with open(config_path, 'r') as f:
        content = f.read()

    new_line = f'SEASON_FEATURES = {repr(season_features_dict)}'

    if 'SEASON_FEATURES = {}' in content:
        content = content.replace('SEASON_FEATURES = {}', new_line)
    elif 'SEASON_FEATURES = ' in content:
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith('SEASON_FEATURES = '):
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    else:
        content += f'\n{new_line}\n'

    with open(config_path, 'w') as f:
        f.write(content)
    print(f'  Updated SEASON_FEATURES in config.py: {season_features_dict}')


def main():
    args = parse_args()

    if not args.force and check_outputs_exist():
        print('Step 2 outputs already exist. Use --force to recompute.')
        return

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_pcc_rows = []
    season_features = {}

    for season in config.SEASON_ORDER:
        print(f'\n=== {season.upper()} ===')
        raw_path = os.path.join(config.SEASONS_DIR, season, 'raw.csv')
        if not os.path.exists(raw_path):
            print(f'  ERROR: {raw_path} not found. Run step1 first.')
            sys.exit(1)

        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        print(f'  Loaded: {df.shape}')

        # Drop rows where WS is NaN for PCC computation
        df_valid = df.dropna(subset=[config.TARGET_VAR])
        print(f'  Valid WS rows: {len(df_valid)}')

        pcc_results = compute_pcc(df_valid, target=config.TARGET_VAR,
                                   candidates=config.ALL_METEO_VARS)

        selected_vars = [r['variable'] for r in pcc_results if r['selected']]
        print(f'  Selected meteo variables: {selected_vars}')

        # Compare with paper expectations
        expected = config.EXPECTED_FEATURES.get(season, [])
        if set(selected_vars) != set(expected):
            print(f'  WARNING: Expected {expected}, got {selected_vars}. '
                  f'Proceeding with computed selection.')
        else:
            print(f'  OK: Selection matches paper expectation: {expected}')

        season_features[season] = selected_vars

        # Record PCC results
        for r in pcc_results:
            all_pcc_rows.append({
                'Season': season,
                'Variable': r['variable'],
                'PCC_r': round(r['r'], 4) if not np.isnan(r['r']) else np.nan,
                'p_value': r['p'],
                'Selected': r['selected'],
            })

        # Build feature dataframe: selected meteo vars + WS (last)
        feature_cols = selected_vars + [config.TARGET_VAR]
        feat_df = df[feature_cols].dropna()
        print(f'  Feature df shape after dropna: {feat_df.shape}')
        print(f'  Columns: {list(feat_df.columns)}')

        train, val, test = split_dataframe(feat_df)
        print(f'  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

        out_dir = os.path.join(config.SEASONS_DIR, season)
        train.to_csv(os.path.join(out_dir, 'train.csv'))
        val.to_csv(os.path.join(out_dir, 'val.csv'))
        test.to_csv(os.path.join(out_dir, 'test.csv'))

        # test_hidden: WS zeroed out
        test_hidden = test.copy()
        test_hidden[config.TARGET_VAR] = 0.0
        test_hidden.to_csv(os.path.join(out_dir, 'test_hidden.csv'))

        print(f'  Saved train/val/test/test_hidden to {out_dir}')

    # Save PCC results table
    pcc_df = pd.DataFrame(all_pcc_rows)
    pcc_path = os.path.join(config.RESULTS_DIR, 'pcc_results_all_seasons.csv')
    pcc_df.to_csv(pcc_path, index=False)
    print(f'\nSaved PCC results to {pcc_path}')

    # Update config.py with computed SEASON_FEATURES
    update_config_season_features(season_features)

    print('\nStep 2 complete.')


if __name__ == '__main__':
    main()
