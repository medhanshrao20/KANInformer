"""
STEP 1 — Load raw CIMIS hourly CSV, convert units, split into 4 seasons,
and save each season's raw.csv.

Run independently:  python step1_prepare_seasons.py [--force]
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Overwrite existing outputs')
    return p.parse_args()


def check_outputs_exist():
    for season in config.SEASON_ORDER:
        path = os.path.join(config.SEASONS_DIR, season, 'raw.csv')
        if not os.path.exists(path):
            return False
    return True


def load_raw_csv(path):
    print(f'Loading {path} ...')
    # Read with header; CIMIS files often have multi-row headers — try first
    df = pd.read_csv(path, low_memory=False)
    print(f'  Raw shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    return df


def clean_and_rename(df):
    """Drop QC columns and rename to standard names."""
    cols = list(df.columns)

    # Identify data vs QC columns.
    # CIMIS layout: Date, Hour (PST), then pairs of (data, QC) columns.
    # QC columns contain 'Qc' or 'qc' in their name, or follow a pattern.
    keep = []
    for c in cols:
        lower = c.strip().lower()
        # Keep if not a QC column
        if 'qc' not in lower:
            keep.append(c)

    df = df[keep].copy()
    print(f'  After dropping QC columns: {list(df.columns)}')

    # Rename columns to standard names
    # Expected order after dropping QC: Date, Hour, ET, PCP, SR, VP, AT, RH, DPT, WS, WD, ST
    expected_count = 12
    if len(df.columns) < expected_count:
        print(f'  WARNING: Expected {expected_count} columns, got {len(df.columns)}')

    rename_map = {}
    col_list = list(df.columns)

    # Map by position (robust to varying column names)
    standard_names = ['Date', 'Hour', 'ET', 'PCP', 'SR', 'VP', 'AT', 'RH', 'DPT', 'WS', 'WD', 'ST']
    for i, std in enumerate(standard_names):
        if i < len(col_list):
            rename_map[col_list[i]] = std

    df = df.rename(columns=rename_map)
    print(f'  Renamed columns: {list(df.columns)}')
    return df


def parse_datetime_index(df):
    """Parse Date column and set as index."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    df = df.drop(columns=['Hour'], errors='ignore')
    return df


def convert_to_numeric(df):
    """Convert all data columns to numeric; coerce errors to NaN."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def convert_units(df):
    """Convert WS from mph to m/s."""
    if 'WS' in df.columns:
        df['WS'] = df['WS'] * 0.44704
        print('  WS converted from mph to m/s')
    return df


def slice_season(df, start_date, end_date):
    """Slice dataframe by date range (inclusive)."""
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df.loc[mask].copy()


def main():
    args = parse_args()

    if not args.force and check_outputs_exist():
        print('Step 1 outputs already exist. Use --force to recompute.')
        return

    if not os.path.exists(config.RAW_CSV):
        print(f'ERROR: {config.RAW_CSV} not found.')
        print('Please place the CIMIS hourly.csv file in the data/ folder.')
        sys.exit(1)

    df = load_raw_csv(config.RAW_CSV)
    df = clean_and_rename(df)
    df = convert_to_numeric(df)
    df = convert_units(df)
    df = parse_datetime_index(df)

    # Keep only the 10 meteorological variables
    meteo_cols = config.ALL_METEO_VARS + [config.TARGET_VAR]
    available = [c for c in meteo_cols if c in df.columns]
    missing = [c for c in meteo_cols if c not in df.columns]
    if missing:
        print(f'  WARNING: Missing columns: {missing}')
    df = df[available]

    print(f'\nFull dataset shape after cleaning: {df.shape}')
    print(f'Date range: {df.index.min()} to {df.index.max()}')

    print('\n--- Splitting into seasons ---')
    for season in config.SEASON_ORDER:
        start, end = config.SEASON_DATES[season]
        season_df = slice_season(df, start, end)

        out_dir = os.path.join(config.SEASONS_DIR, season)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'raw.csv')
        season_df.to_csv(out_path)

        print(f'  {season.capitalize():8s}: {len(season_df):5d} rows  '
              f'({start} to {end})  -> {out_path}')
        print(f'    WS stats: min={season_df["WS"].min():.3f}, '
              f'max={season_df["WS"].max():.3f}, '
              f'mean={season_df["WS"].mean():.3f} m/s')
        print(f'    NaN counts: {season_df.isna().sum().to_dict()}')

    print('\nStep 1 complete.')


if __name__ == '__main__':
    main()
