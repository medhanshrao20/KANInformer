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
    """Drop QC columns, metadata columns, and rename CIMIS names to standard names."""
    # Drop QC columns (exact name 'qc' or contains 'qc' as standalone QC field)
    cols = list(df.columns)
    keep = []
    for c in cols:
        lower = c.strip().lower()
        if lower == 'qc' or lower.startswith('qc.'):
            continue
        keep.append(c)
    df = df[keep].copy()

    # Drop CIMIS station metadata (not used in the paper pipeline)
    drop_meta = ['Stn Id', 'Stn Name', 'CIMIS Region', 'Jul']
    df = df.drop(columns=[c for c in drop_meta if c in df.columns], errors='ignore')
    print(f'  After dropping QC + metadata: {list(df.columns)}')

    # Rename by actual CIMIS column names (not by position)
    cimis_rename = {
        'Date': 'Date',
        'Hour (PST)': 'Hour',
        'ETo (in)': 'ET',
        'Precip (in)': 'PCP',
        'Sol Rad (Ly/day)': 'SR',
        'Vap Pres (mBars)': 'VP',
        'Air Temp (F)': 'AT',
        'Rel Hum (%)': 'RH',
        'Dew Point (F)': 'DPT',
        'Wind Speed (mph)': 'WS',
        'Wind Dir (0-360)': 'WD',
        'Soil Temp (F)': 'ST',
    }
    df = df.rename(columns=cimis_rename)
    print(f'  Renamed columns: {list(df.columns)}')
    return df


def parse_datetime_index(df):
    """Combine Date + Hour into hourly datetime index."""
    # CIMIS Date: MM/DD/YYYY  e.g. 12/1/2020
    # CIMIS Hour: 0100, 0200, ... (HHMM, 24h) — may be read as int 100, 200
    def format_hour(h):
        s = str(h).strip().split('.')[0]  # 100.0 -> "100"
        if not s.isdigit():
            return s
        val = int(s)
        if val == 2400:
            return '0000'  # midnight next day — date adjustment handled below
        return str(val).zfill(4)

    hour_str = df['Hour'].apply(format_hour)
    datetime_str = df['Date'].astype(str).str.strip() + ' ' + hour_str
    dt = pd.to_datetime(datetime_str, format='%m/%d/%Y %H%M', errors='coerce')

    # CIMIS sometimes uses 2400 for end-of-day; roll to next calendar day 00:00
    hour_raw = df['Hour'].apply(lambda h: str(h).strip().split('.')[0])
    is_2400 = hour_raw == '2400'
    if is_2400.any():
        dt.loc[is_2400] = pd.to_datetime(
            df.loc[is_2400, 'Date'].astype(str).str.strip(),
            format='%m/%d/%Y',
            errors='coerce',
        ) + pd.Timedelta(days=1)

    n_bad = dt.isna().sum()
    if n_bad > 0:
        print(f'  WARNING: {n_bad} rows failed strict datetime parse, retrying flexible parse')
        bad = dt.isna()
        dt.loc[bad] = pd.to_datetime(datetime_str.loc[bad], errors='coerce')

    df = df.copy()
    df['datetime'] = dt
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    df = df.drop(columns=['Date', 'Hour'], errors='ignore')
    df.index.name = 'Date'
    print(f'  Parsed {len(df)} hourly timestamps')
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
    """Slice dataframe by date range, inclusive of all hours on the last day."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df.index >= start) & (df.index <= end)
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
    df = parse_datetime_index(df)   # must run before numeric conversion (Date/Hour are strings)
    df = convert_to_numeric(df)
    df = convert_units(df)

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

        if len(season_df) == 0:
            print(f'  ERROR: {season} has 0 rows — check Date/Hour parsing in hourly.csv')
            sys.exit(1)

    print('\nStep 1 complete.')


if __name__ == '__main__':
    main()
