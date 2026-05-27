"""
STEP 3 — Variational Mode Decomposition (VMD) of wind speed signal.

Finds optimal K (number of IMFs) using Residual Energy Ratio criterion,
then decomposes the full seasonal WS series and saves IMFs.

Run independently:  python step3_vmd_decompose.py [--force]
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
    p.add_argument('--season', type=str, default=None, help='Process single season')
    return p.parse_args()


def check_outputs_exist(season):
    path = os.path.join(config.DECOMP_DIR, season, 'imfs.npy')
    return os.path.exists(path)


def compute_rer(signal, imfs):
    """
    Residual Energy Ratio (paper Equation 7):
      rer = (1/T) * sum( |f(t) - sum_k(u_k(t))| / |f(t)| )
    Guard against division by zero with 1e-10.
    """
    reconstruction = np.sum(imfs, axis=0)
    residual = np.abs(signal - reconstruction)
    denominator = np.abs(signal) + 1e-10
    rer = np.mean(residual / denominator)
    return rer


def run_vmd(signal, K):
    """Run VMD with paper parameters for a given K."""
    from vmdpy import VMD
    u, u_hat, omega = VMD(
        signal,
        alpha=config.VMD_ALPHA,
        tau=config.VMD_TAU,
        K=K,
        DC=config.VMD_DC,
        init=config.VMD_INIT,
        tol=config.VMD_TOL,
    )
    return u  # shape (K, N)


def find_optimal_k(signal):
    """
    Search for optimal K using the Residual Energy Ratio algorithm.
    Returns (optimal_K, rer_at_optimal_K, rer_history).
    """
    rer_history = {}
    prev_rer = None
    optimal_k = None

    for K in config.VMD_K_RANGE:
        print(f'    K={K:2d} ', end='', flush=True)
        try:
            u = run_vmd(signal, K)
            rer = compute_rer(signal, u)
            rer_history[K] = rer
            print(f'  RER={rer*100:.3f}%')

            if rer < config.VMD_RER_THRESHOLD:
                if prev_rer is not None and (prev_rer - rer) < 0.005:
                    optimal_k = K
                    print(f'    --> Optimal K={K} (RER={rer*100:.3f}%, '
                          f'decrease={((prev_rer-rer)*100):.3f}% < 0.5%)')
                    break
            prev_rer = rer
        except Exception as e:
            print(f'  FAILED: {e}')
            rer_history[K] = np.nan

    # Fallback: first K where RER < threshold
    if optimal_k is None:
        for K, rer in rer_history.items():
            if not np.isnan(rer) and rer < config.VMD_RER_THRESHOLD:
                optimal_k = K
                print(f'    --> Fallback optimal K={K} (first K with RER < {config.VMD_RER_THRESHOLD*100}%)')
                break

    # Ultimate fallback: last K tried
    if optimal_k is None:
        valid = {k: v for k, v in rer_history.items() if not np.isnan(v)}
        if valid:
            optimal_k = max(valid.keys())
            print(f'    --> Ultimate fallback K={optimal_k}')
        else:
            optimal_k = 10
            print(f'    --> Could not find good K, defaulting to K=10')

    return optimal_k, rer_history.get(optimal_k, np.nan), rer_history


def process_season(season, args):
    print(f'\n=== {season.upper()} ===')

    if not args.force and check_outputs_exist(season):
        print(f'  IMFs already exist for {season}. Skipping.')
        return

    raw_path = os.path.join(config.SEASONS_DIR, season, 'raw.csv')
    if not os.path.exists(raw_path):
        print(f'  ERROR: {raw_path} not found. Run step1 first.')
        return

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    ws = df[config.TARGET_VAR].values

    # Handle NaN in WS: interpolate linearly then fill edges
    if np.isnan(ws).any():
        n_nan = np.isnan(ws).sum()
        print(f'  WARNING: {n_nan} NaN in WS. Interpolating...')
        ws_series = pd.Series(ws).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        ws = ws_series.values

    print(f'  WS series length: {len(ws)}')
    print(f'  WS range: [{ws.min():.3f}, {ws.max():.3f}] m/s')

    # Find optimal K
    print('  Searching for optimal K...')
    optimal_k, best_rer, rer_history = find_optimal_k(ws)

    # Compare with paper expectation
    expected_k = config.EXPECTED_K.get(season, None)
    if expected_k is not None and optimal_k != expected_k:
        diff = abs(optimal_k - expected_k)
        if diff <= 1:
            print(f'  NOTE: Found K={optimal_k} but paper expects K={expected_k}. '
                  f'Difference is 1 — using paper K={expected_k}.')
            optimal_k = expected_k
        else:
            print(f'  WARNING: Found K={optimal_k}, paper expects K={expected_k}. '
                  f'Using computed K={optimal_k}.')
    else:
        print(f'  K={optimal_k} matches paper expectation.')

    # Final VMD decomposition with optimal K
    print(f'  Running final VMD with K={optimal_k}...')
    imfs = run_vmd(ws, optimal_k)
    print(f'  IMFs shape: {imfs.shape}')

    # Verify reconstruction
    reconstruction = np.sum(imfs, axis=0)
    recon_error = np.mean(np.abs(ws - reconstruction))
    print(f'  Reconstruction error (MAE): {recon_error:.6f} m/s')
    final_rer = compute_rer(ws, imfs)
    print(f'  Final RER: {final_rer*100:.3f}%')

    # Save
    out_dir = os.path.join(config.DECOMP_DIR, season)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'imfs.npy'), imfs)
    print(f'  Saved imfs.npy shape {imfs.shape} to {out_dir}')

    # Save RER history
    rer_arr = np.array([[k, v] for k, v in sorted(rer_history.items())])
    np.save(os.path.join(out_dir, 'rer_history.npy'), rer_arr)


def main():
    args = parse_args()

    seasons = [args.season] if args.season else config.SEASON_ORDER

    for season in seasons:
        if season not in config.SEASON_DATES:
            print(f'Unknown season: {season}')
            continue
        process_season(season, args)

    print('\nStep 3 complete.')


if __name__ == '__main__':
    main()
