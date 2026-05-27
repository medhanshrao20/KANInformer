"""
STEP 6 — Empirical Wavelet Transform (EWT) decomposition of fused IMF.

Applies EWT to the fused high-SE component from Step 5.
N_EWT = 8 sub-modes (fixed per paper).

Run independently:  python step6_ewt_decompose.py [--force]
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    p.add_argument('--season', type=str, default=None)
    return p.parse_args()


def check_outputs_exist(season):
    return os.path.exists(os.path.join(config.DECOMP_DIR, season, 'ewt_modes.npy'))


def apply_ewt(signal, n_modes=8):
    """
    Apply EWT to signal using ewtpy.
    Returns ewt_modes of shape (n_modes, N).
    Pads with zeros if fewer than n_modes are returned.
    """
    import ewtpy

    try:
        ewt, mfb, boundaries = ewtpy.EWT1D(signal, N=n_modes)
        # ewt shape from ewtpy is (len(signal), num_modes_found)
        print(f'    EWT returned shape: {ewt.shape}')
        print(f'    Number of frequency boundaries: {len(boundaries)}')

        n_found = ewt.shape[1] if ewt.ndim == 2 else 1
        if ewt.ndim == 1:
            ewt = ewt.reshape(-1, 1)

        ewt_modes = ewt.T  # (n_found, N)

        if n_found < n_modes:
            print(f'    WARNING: EWT returned {n_found} modes, expected {n_modes}. '
                  f'Padding with zeros.')
            N = signal.shape[0]
            pad = np.zeros((n_modes - n_found, N))
            ewt_modes = np.vstack([ewt_modes, pad])
        elif n_found > n_modes:
            print(f'    NOTE: EWT returned {n_found} modes, truncating to {n_modes}.')
            ewt_modes = ewt_modes[:n_modes, :]

        return ewt_modes  # (n_modes, N)

    except Exception as e:
        print(f'    ERROR in EWT: {e}')
        print(f'    Falling back to zero-padded modes.')
        N = len(signal)
        return np.zeros((n_modes, N))


def process_season(season, args):
    print(f'\n=== {season.upper()} ===')

    if not args.force and check_outputs_exist(season):
        print(f'  EWT modes already exist for {season}. Skipping.')
        return

    fused_path = os.path.join(config.DECOMP_DIR, season, 'fused_imf.npy')
    if not os.path.exists(fused_path):
        print(f'  ERROR: {fused_path} not found. Run step5 first.')
        return

    fused_imf = np.load(fused_path)
    print(f'  fused_imf shape: {fused_imf.shape}')

    # Apply EWT
    print(f'  Applying EWT with N={config.N_EWT} modes...')
    ewt_modes = apply_ewt(fused_imf, n_modes=config.N_EWT)
    print(f'  EWT modes shape: {ewt_modes.shape}')

    # Verify reconstruction quality
    reconstruction = np.sum(ewt_modes, axis=0)
    recon_error = np.mean(np.abs(fused_imf - reconstruction))
    print(f'  EWT reconstruction MAE: {recon_error:.6f}')

    out_dir = os.path.join(config.DECOMP_DIR, season)
    np.save(os.path.join(out_dir, 'ewt_modes.npy'), ewt_modes)
    print(f'  Saved ewt_modes.npy shape {ewt_modes.shape} to {out_dir}')


def main():
    args = parse_args()
    seasons = [args.season] if args.season else config.SEASON_ORDER

    for season in seasons:
        if season not in config.SEASON_DATES:
            continue
        process_season(season, args)

    print('\nStep 6 complete.')


if __name__ == '__main__':
    main()
