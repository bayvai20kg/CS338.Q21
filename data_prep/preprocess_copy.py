"""
preprocess_copy.py
==================
Preprocessing pipeline chạy trên BẢN SAO dataset (dataset_copy/).
Output lưu vào dataset_processed/.

Đây là bản độc lập, không ảnh hưởng đến dataset gốc.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
SRC_FILE   = 'dataset_copy/CLN_SG_V2.xlsx'   # bản sao dataset
OUTPUT_DIR = 'dataset_processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'Nongdo_Jartest_PAC'

FEATURE_COLS = [
    'Doduc_vao_nha_may',
    'pH_Song_SG',
    'pH_vao_nha_may',
    'Man_song_saigon',
    'NH3_song_sai_gon',
    'Mn_vao_nha_may',
    'Fe_vao_nha_may',
    'DO_vao_nha_may',
    'Mau_vao_nha_may_BK',
    'SS_vao_nha_may',
    'Thetich_Jartest_PAC',
]

LAG_COLS    = ['Doduc_vao_nha_may', 'Man_song_saigon', 'pH_Song_SG',
               'Mn_vao_nha_may', TARGET]
LAGS        = [1, 2, 3, 7, 14]
ROLLING_COLS = ['Doduc_vao_nha_may', 'Man_song_saigon', 'Mn_vao_nha_may', TARGET]


def run_preprocessing(verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    log("=" * 58)
    log("  PREPROCESSING (bản sao) → dataset_processed/")
    log("=" * 58)

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    df = pd.read_excel(SRC_FILE)
    df.columns = df.columns.str.strip()
    df['Ngay'] = pd.to_datetime(df['Ngay'])
    df = df.sort_values('Ngay').reset_index(drop=True).set_index('Ngay')
    log(f"\n[1] Loaded: {df.shape}")

    # ── STEP 2: Drop cột >50% missing ─────────────────────────────────────────
    missing_pct = df.isnull().sum() / len(df) * 100
    drop_cols = missing_pct[missing_pct > 50].index.tolist()
    df = df.drop(columns=drop_cols)
    log(f"[2] Drop {len(drop_cols)} cols (>50% missing): {drop_cols}")

    # ── STEP 3: Chọn features + target ────────────────────────────────────────
    df_work = df[FEATURE_COLS + [TARGET]].copy()
    log(f"[3] Features: {len(FEATURE_COLS)} + 1 target")

    # ── STEP 4: Missing value handling ────────────────────────────────────────
    df_work = df_work.dropna(subset=[TARGET])
    df_work[FEATURE_COLS] = (
        df_work[FEATURE_COLS]
        .interpolate(method='linear', limit_direction='both')
        .ffill().bfill()
    )
    log(f"[4] After drop target NaN + interpolate: {len(df_work)} rows, "
        f"{df_work.isnull().sum().sum()} missing left")

    # ── STEP 5: Outlier clipping (IQR x3) ────────────────────────────────────
    for col in FEATURE_COLS:
        Q1, Q3 = df_work[col].quantile(0.25), df_work[col].quantile(0.75)
        IQR = Q3 - Q1
        df_work[col] = df_work[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)
    log("[5] Outlier clipping done")

    # ── STEP 6: Feature Engineering ───────────────────────────────────────────
    df_fe = df_work.copy()

    # Temporal
    df_fe['month']        = df_fe.index.month
    df_fe['day_of_year']  = df_fe.index.dayofyear
    df_fe['week']         = df_fe.index.isocalendar().week.astype(int)
    df_fe['sin_month']    = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['cos_month']    = np.cos(2 * np.pi * df_fe['month'] / 12)
    df_fe['sin_doy']      = np.sin(2 * np.pi * df_fe['day_of_year'] / 365)
    df_fe['cos_doy']      = np.cos(2 * np.pi * df_fe['day_of_year'] / 365)
    df_fe['is_dry_season']= df_fe['month'].isin([11,12,1,2,3,4]).astype(int)

    # Lag features
    for col in LAG_COLS:
        for lag in LAGS:
            df_fe[f'{col}_lag{lag}'] = df_fe[col].shift(lag)

    # Rolling stats
    for col in ROLLING_COLS:
        df_fe[f'{col}_roll7_mean']  = df_fe[col].shift(1).rolling(7,  min_periods=1).mean()
        df_fe[f'{col}_roll7_std']   = df_fe[col].shift(1).rolling(7,  min_periods=1).std().fillna(0)
        df_fe[f'{col}_roll14_mean'] = df_fe[col].shift(1).rolling(14, min_periods=1).mean()

    # Log transform
    df_fe['Doduc_log']      = np.log1p(df_fe['Doduc_vao_nha_may'])
    df_fe['Doduc_log_lag1'] = df_fe['Doduc_log'].shift(1)

    before = len(df_fe)
    df_fe = df_fe.dropna()
    log(f"[6] Feature engineering: {df_fe.shape[1]-1} features, "
        f"drop {before - len(df_fe)} lag rows → {len(df_fe)} rows")

    # ── STEP 7: Time-based split ──────────────────────────────────────────────
    train = df_fe[df_fe.index.year <= 2020]
    val   = df_fe[df_fe.index.year == 2021]
    test  = df_fe[df_fe.index.year == 2022]
    log(f"[7] Split → Train:{len(train)} Val:{len(val)} Test:{len(test)}")

    # ── STEP 8: Min-Max scaling (fit on train) ────────────────────────────────
    all_cols = [c for c in df_fe.columns]
    scale_params = {}
    for col in all_cols:
        mn = float(train[col].min())
        mx = float(train[col].max())
        scale_params[col] = {'min': mn, 'max': mx}

    def scale(df_in):
        df_out = df_in.copy()
        for col, p in scale_params.items():
            if col in df_out.columns:
                rng = p['max'] - p['min'] if p['max'] != p['min'] else 1.0
                df_out[col] = (df_out[col] - p['min']) / rng
        return df_out

    train_s = scale(train)
    val_s   = scale(val)
    test_s  = scale(test)
    log("[8] Min-Max scaling done (fit on train only)")

    # ── STEP 9: Save ──────────────────────────────────────────────────────────
    feature_cols = [c for c in df_fe.columns if c != TARGET]

    train_s.to_csv(f'{OUTPUT_DIR}/train.csv')
    val_s.to_csv(f'{OUTPUT_DIR}/val.csv')
    test_s.to_csv(f'{OUTPUT_DIR}/test.csv')
    df_fe.to_csv(f'{OUTPUT_DIR}/processed_full.csv')

    with open(f'{OUTPUT_DIR}/scaler_params.json', 'w', encoding='utf-8') as f:
        json.dump(scale_params, f, indent=2, ensure_ascii=False)

    meta = {
        'target': TARGET,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'train_rows': len(train_s),
        'val_rows': len(val_s),
        'test_rows': len(test_s),
        'date_range': {
            'train': [str(train.index.min().date()), str(train.index.max().date())],
            'val':   [str(val.index.min().date()),   str(val.index.max().date())],
            'test':  [str(test.index.min().date()),  str(test.index.max().date())],
        },
    }
    with open(f'{OUTPUT_DIR}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log(f"\n[9] Saved to {OUTPUT_DIR}/")
    log(f"    train.csv: {train_s.shape}  val.csv: {val_s.shape}  test.csv: {test_s.shape}")
    log("=" * 58)
    log("  PREPROCESSING DONE!")
    log("=" * 58)

    return meta


if __name__ == '__main__':
    run_preprocessing()
