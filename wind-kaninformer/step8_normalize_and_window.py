"""
STEP 8 — Normalize feature matrix and create sliding windows.

MinMaxScaler fitted on training rows only.
Sliding windows: X=(7, N_heng), Y=(3,) for multi-step forecasting.

Run independently:  python step8_normalize_and_window.py [--force]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    p.add_argument('--season', type=str, default=None)
    return p.parse_args()


def check_outputs_exist(season):
    for split in ['train', 'val', 'test']:
        for xy in ['X', 'Y']:
            path = os.path.join(config.PROCESSED_DIR, f'{season}_{xy}_{split}.npy')
            if not os.path.exists(path):
                return False
    return True


def split_dataframe(data):
    """Chronological 8:1:1 split on numpy array."""
    n = len(data)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def fit_scalers_per_column(train_data):
    """Fit one MinMaxScaler per column using only training data."""
    n_cols = train_data.shape[1]
    scalers = []
    for c in range(n_cols):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data[:, c].reshape(-1, 1))
        scalers.append(scaler)
    return scalers


def transform_with_scalers(data, scalers):
    """Apply fitted scalers column-by-column."""
    transformed = np.zeros_like(data, dtype=np.float32)
    for c, scaler in enumerate(scalers):
        transformed[:, c] = scaler.transform(
            data[:, c].reshape(-1, 1)
        ).flatten()
    return transformed


def create_sliding_windows(data, n_step, n_out):
    """
    Create X/Y pairs from normalized data.
    X[i] = data[i:i+n_step, :]          shape (n_step, N_heng)
    Y[i] = data[i+n_step:i+n_step+n_out, -1]  shape (n_out,)
    """
    n = len(data)
    n_samples = n - n_step - n_out + 1
    if n_samples <= 0:
        raise ValueError(f'Not enough data: len={n}, n_step={n_step}, n_out={n_out}')

    n_heng = data.shape[1]
    X = np.zeros((n_samples, n_step, n_heng), dtype=np.float32)
    Y = np.zeros((n_samples, n_out), dtype=np.float32)

    for i in range(n_samples):
        X[i] = data[i:i + n_step, :]
        Y[i] = data[i + n_step:i + n_step + n_out, -1]

    return X, Y


def process_season(season, args):
    print(f'\n=== {season.upper()} ===')

    if not args.force and check_outputs_exist(season):
        print(f'  Windows already exist for {season}. Skipping.')
        return

    feat_path = os.path.join(config.SEASONS_DIR, season, 'feature_matrix.csv')
    if not os.path.exists(feat_path):
        print(f'  ERROR: {feat_path} not found. Run step7 first.')
        return

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    data = df.values.astype(np.float32)
    print(f'  Feature matrix shape: {data.shape}')
    print(f'  N_heng: {data.shape[1]}')

    # Split
    train_raw, val_raw, test_raw = split_dataframe(data)
    print(f'  Split sizes — train: {len(train_raw)}, val: {len(val_raw)}, test: {len(test_raw)}')

    # Fit scalers on training data only
    scalers = fit_scalers_per_column(train_raw)
    print(f'  Fitted {len(scalers)} column scalers on training data')

    # Transform all splits
    train_norm = transform_with_scalers(train_raw, scalers)
    val_norm = transform_with_scalers(val_raw, scalers)
    test_norm = transform_with_scalers(test_raw, scalers)

    # Verify WS scaler (last scaler)
    ws_scaler = scalers[-1]
    print(f'  WS scaler range: [{ws_scaler.data_min_[0]:.4f}, {ws_scaler.data_max_[0]:.4f}]')

    # Save scalers
    scaler_path = os.path.join(config.SEASONS_DIR, season, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f'  Saved {len(scalers)} scalers to {scaler_path}')

    # Create sliding windows
    n_step = config.N_STEP
    n_out = config.N_OUT

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    for split_name, split_data in [('train', train_norm),
                                    ('val', val_norm),
                                    ('test', test_norm)]:
        X, Y = create_sliding_windows(split_data, n_step, n_out)
        X_path = os.path.join(config.PROCESSED_DIR, f'{season}_X_{split_name}.npy')
        Y_path = os.path.join(config.PROCESSED_DIR, f'{season}_Y_{split_name}.npy')
        np.save(X_path, X)
        np.save(Y_path, Y)
        print(f'  {split_name:5s}: X={X.shape}, Y={Y.shape}')


def main():
    args = parse_args()
    seasons = [args.season] if args.season else config.SEASON_ORDER

    for season in seasons:
        if season not in config.SEASON_DATES:
            continue
        process_season(season, args)

    print('\nStep 8 complete.')


if __name__ == '__main__':
    main()
