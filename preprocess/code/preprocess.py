# ============================================================
# PREPROCESSING — CLN_SG_V2 (Water Quality Dataset)
# Target: Nongdo_Jartest_PAC  (PAC concentration — daily)
# Task  : Time-Series Forecasting with LNN / LTC
# ============================================================
# Output files:
#   dataset/processed_full.csv   — toàn bộ sau khi clean
#   dataset/train.csv            — train  (2017-2020)
#   dataset/val.csv              — val    (2021)
#   dataset/test.csv             — test   (2022)
#   dataset/scaler_params.json   — min/max của từng cột (để inverse transform)
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── Style chung ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#3a3d5c',   'axes.labelcolor': '#c8cce8',
    'xtick.color': '#8890c8',      'ytick.color': '#8890c8',
    'text.color': '#e8eaf6',       'grid.color': '#2a2d4a',
    'grid.linewidth': 0.6,         'figure.dpi': 120,
    'savefig.dpi': 150,            'savefig.facecolor': '#0f1117',
    'axes.titlecolor': '#a0c4ff',  'axes.titleweight': 'bold',
    'font.size': 10,
})
PALETTE = ['#00b4d8', '#f72585', '#7209b7', '#3a86ff', '#ff006e',
           '#8338ec', '#06d6a0', '#ffbe0b', '#fb5607', '#4cc9f0']
ACCENT = '#a0c4ff'
OUTPUT_DIR = 'dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & strip column names
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  PREPROCESSING — CLN_SG_V2  |  Target: Nongdo_Jartest_PAC")
print("=" * 62)

df_raw = pd.read_excel('dataset/CLN_SG_V2.xlsx')
df_raw.columns = df_raw.columns.str.strip()          # xóa trailing spaces
df_raw['Ngay'] = pd.to_datetime(df_raw['Ngay'])
df_raw = df_raw.sort_values('Ngay').reset_index(drop=True).set_index('Ngay')

print(f"\n[1] Raw shape: {df_raw.shape}  |  {df_raw.index.min().date()} → {df_raw.index.max().date()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Chọn cột: loại bỏ cột quá nhiều missing (> 50%)
# ══════════════════════════════════════════════════════════════════════════════
missing_pct = df_raw.isnull().sum() / len(df_raw) * 100
DROP_COLS = missing_pct[missing_pct > 50].index.tolist()
df = df_raw.drop(columns=DROP_COLS)
print(f"\n[2] Bỏ {len(DROP_COLS)} cột missing > 50%: {DROP_COLS}")
print(f"    Còn lại: {df.shape[1]} cột")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Chọn tập features phù hợp cho bài toán dự báo PAC
#           (chỉ dùng features đầu vào — không leak thông tin đầu ra)
# ══════════════════════════════════════════════════════════════════════════════
TARGET = 'Nongdo_Jartest_PAC'

# Features: các chỉ tiêu nước thô (sông / đầu vào nhà máy)
FEATURE_COLS = [
    'Doduc_vao_nha_may',    # Độ đục đầu vào (NTU) — quan trọng nhất với PAC
    'pH_Song_SG',           # pH sông Sài Gòn
    'pH_vao_nha_may',       # pH tại đầu vào nhà máy
    'Man_song_saigon',      # Độ mặn sông (TDS)
    'NH3_song_sai_gon',     # Amoniac nước thô
    'Mn_vao_nha_may',       # Mangan đầu vào
    'Fe_vao_nha_may',       # Sắt đầu vào
    'DO_vao_nha_may',       # Oxy hòa tan đầu vào
    'Mau_vao_nha_may_BK',   # Màu nước thô
    'SS_vao_nha_may',       # Chất rắn lơ lửng đầu vào
    'Thetich_Jartest_PAC',  # Thể tích PAC trong Jar test (liên quan trực tiếp)
]

df_work = df[FEATURE_COLS + [TARGET]].copy()
print(f"\n[3] Feature set: {len(FEATURE_COLS)} features + 1 target")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Xử lý missing values
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Missing values trước xử lý:")
for col in df_work.columns:
    n = df_work[col].isnull().sum()
    if n > 0:
        print(f"    {col:35s}: {n:4d} ({n/len(df_work)*100:.1f}%)")

# 4a — Với cột target: chỉ giữ các ngày có giá trị PAC thực (không impute target)
df_work = df_work.dropna(subset=[TARGET])
print(f"\n    Sau khi drop target NaN: {len(df_work)} rows còn lại")

# 4b — Với features: dùng linear interpolation (time-series phù hợp)
#      sau đó fill forward/backward cho đầu/cuối chuỗi
df_work[FEATURE_COLS] = (
    df_work[FEATURE_COLS]
    .interpolate(method='linear', limit_direction='both')
    .ffill()
    .bfill()
)

# 4c — Kiểm tra còn missing không
remaining = df_work.isnull().sum().sum()
print(f"    Missing còn lại sau interpolation: {remaining}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Loại bỏ outliers cực đoan (IQR × 3 — gentle clipping)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Clipping outliers (IQR × 3):")
for col in FEATURE_COLS:
    Q1 = df_work[col].quantile(0.25)
    Q3 = df_work[col].quantile(0.75)
    IQR = Q3 - Q1
    lo = Q1 - 3 * IQR
    hi = Q3 + 3 * IQR
    n_out = ((df_work[col] < lo) | (df_work[col] > hi)).sum()
    if n_out > 0:
        print(f"    {col:35s}: clip {n_out} outliers → [{lo:.3f}, {hi:.3f}]")
    df_work[col] = df_work[col].clip(lower=lo, upper=hi)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Feature Engineering (Time-Series specific)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Feature Engineering...")

df_fe = df_work.copy()

# 6a — Temporal features
df_fe['month']       = df_fe.index.month
df_fe['day_of_year'] = df_fe.index.dayofyear
df_fe['week']        = df_fe.index.isocalendar().week.astype(int)
# Fourier encoding for seasonality (12-month cycle)
df_fe['sin_month']   = np.sin(2 * np.pi * df_fe['month'] / 12)
df_fe['cos_month']   = np.cos(2 * np.pi * df_fe['month'] / 12)
df_fe['sin_doy']     = np.sin(2 * np.pi * df_fe['day_of_year'] / 365)
df_fe['cos_doy']     = np.cos(2 * np.pi * df_fe['day_of_year'] / 365)
# Mùa khô / mưa (đặc trưng miền Nam VN)
df_fe['is_dry_season'] = df_fe['month'].isin([11,12,1,2,3,4]).astype(int)

# 6b — Lag features cho chỉ tiêu quan trọng nhất
LAG_COLS = ['Doduc_vao_nha_may', 'Man_song_saigon', 'pH_Song_SG',
            'Mn_vao_nha_may', TARGET]
LAGS = [1, 2, 3, 7, 14]

for col in LAG_COLS:
    for lag in LAGS:
        df_fe[f'{col}_lag{lag}'] = df_fe[col].shift(lag)

# 6c — Rolling statistics (7-day và 14-day)
ROLLING_COLS = ['Doduc_vao_nha_may', 'Man_song_saigon', 'Mn_vao_nha_may', TARGET]
for col in ROLLING_COLS:
    df_fe[f'{col}_roll7_mean']  = df_fe[col].shift(1).rolling(7,  min_periods=1).mean()
    df_fe[f'{col}_roll7_std']   = df_fe[col].shift(1).rolling(7,  min_periods=1).std().fillna(0)
    df_fe[f'{col}_roll14_mean'] = df_fe[col].shift(1).rolling(14, min_periods=1).mean()

# 6d — Biến đổi độ đục: log transform (phân phối lệch phải mạnh)
df_fe['Doduc_log'] = np.log1p(df_fe['Doduc_vao_nha_may'])
df_fe['Doduc_log_lag1'] = df_fe['Doduc_log'].shift(1)

# 6e — Drop rows NaN do lag (mất tối đa lag=14 ngày)
before = len(df_fe)
df_fe = df_fe.dropna()
print(f"    Drop {before - len(df_fe)} rows do lag features → còn {len(df_fe)} rows")
print(f"    Tổng số features sau engineering: {df_fe.shape[1] - 1} (không kể target)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Train / Val / Test split (theo thời gian — KHÔNG random)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Train/Val/Test split theo thời gian:")
train = df_fe[df_fe.index.year <= 2020]
val   = df_fe[df_fe.index.year == 2021]
test  = df_fe[df_fe.index.year == 2022]

print(f"    Train : {train.index.min().date()} → {train.index.max().date()}  | {len(train):4d} rows ({len(train)/len(df_fe)*100:.1f}%)")
print(f"    Val   : {val.index.min().date()}   → {val.index.max().date()}   | {len(val):4d} rows ({len(val)/len(df_fe)*100:.1f}%)")
print(f"    Test  : {test.index.min().date()}  → {test.index.max().date()}  | {len(test):4d} rows ({len(test)/len(df_fe)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Normalization (Min-Max scaling fit trên train only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Min-Max Normalization (fit on train only)...")

# Tách features & target
all_feat_cols = [c for c in df_fe.columns if c != TARGET]
SCALE_COLS = all_feat_cols + [TARGET]

# Tính min/max từ train
scale_params = {}
for col in SCALE_COLS:
    col_min = train[col].min()
    col_max = train[col].max()
    scale_params[col] = {'min': col_min, 'max': col_max}

def minmax_scale(df_in, params):
    df_out = df_in.copy()
    for col in params:
        if col in df_out.columns:
            mn = params[col]['min']
            mx = params[col]['max']
            rng = mx - mn if mx != mn else 1.0
            df_out[col] = (df_out[col] - mn) / rng
    return df_out

train_scaled = minmax_scale(train, scale_params)
val_scaled   = minmax_scale(val,   scale_params)
test_scaled  = minmax_scale(test,  scale_params)

# Verify range
target_min_tr = train_scaled[TARGET].min()
target_max_tr = train_scaled[TARGET].max()
print(f"    Target scaled range (train): [{target_min_tr:.4f}, {target_max_tr:.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Lưu files
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Lưu files...")

# Lưu dạng scaled
train_scaled.to_csv(f'{OUTPUT_DIR}/train.csv')
val_scaled.to_csv(f'{OUTPUT_DIR}/val.csv')
test_scaled.to_csv(f'{OUTPUT_DIR}/test.csv')

# Lưu toàn bộ processed (chưa scale) để tham khảo
df_fe.to_csv(f'{OUTPUT_DIR}/processed_full.csv')

# Lưu scale params
scale_params_native = {
    col: {'min': float(v['min']), 'max': float(v['max'])}
    for col, v in scale_params.items()
}
with open(f'{OUTPUT_DIR}/scaler_params.json', 'w', encoding='utf-8') as f:
    json.dump(scale_params_native, f, indent=2, ensure_ascii=False)

# Lưu metadata
meta = {
    'target': TARGET,
    'feature_cols': all_feat_cols,
    'n_features': len(all_feat_cols),
    'train_rows': len(train_scaled),
    'val_rows': len(val_scaled),
    'test_rows': len(test_scaled),
    'date_range': {
        'train': [str(train.index.min().date()), str(train.index.max().date())],
        'val':   [str(val.index.min().date()),   str(val.index.max().date())],
        'test':  [str(test.index.min().date()),  str(test.index.max().date())],
    },
    'lag_features': {'cols': LAG_COLS, 'lags': LAGS},
    'drop_cols_gt50pct': DROP_COLS,
}
with open(f'{OUTPUT_DIR}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"    train.csv          : {len(train_scaled):4d} rows × {train_scaled.shape[1]} cols")
print(f"    val.csv            : {len(val_scaled):4d} rows × {val_scaled.shape[1]} cols")
print(f"    test.csv           : {len(test_scaled):4d} rows × {test_scaled.shape[1]} cols")
print(f"    processed_full.csv : {len(df_fe):4d} rows × {df_fe.shape[1]} cols")
print(f"    scaler_params.json : {len(scale_params)} columns")
print(f"    metadata.json      : saved")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Visualizations kiểm tra preprocessing
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Tạo biểu đồ kiểm tra preprocessing...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Preprocessing Verification — CLN_SG_V2 | Target: Nongdo_Jartest_PAC',
             fontsize=14, fontweight='bold', color='#e8eaf6', y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# Plot 1 — Target distribution: raw vs scaled
ax1 = fig.add_subplot(gs[0, 0])
df_fe[TARGET].hist(bins=40, ax=ax1, color='#00b4d8', edgecolor='none', alpha=0.8)
ax1.set_title('Target: phân phối gốc (mg/L)', fontsize=9)
ax1.set_xlabel('PAC concentration')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
train_scaled[TARGET].hist(bins=40, ax=ax2, color='#06d6a0', edgecolor='none', alpha=0.8)
val_scaled[TARGET].hist(bins=40, ax=ax2, color='#ffbe0b', edgecolor='none', alpha=0.5)
test_scaled[TARGET].hist(bins=40, ax=ax2, color='#f72585', edgecolor='none', alpha=0.5)
ax2.set_title('Target scaled [0,1]', fontsize=9)
ax2.legend(['Train', 'Val', 'Test'], fontsize=7)
ax2.grid(True, alpha=0.3)

# Plot 2 — Time-series splits
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(train.index, train[TARGET], color='#00b4d8', linewidth=0.8, label='Train')
ax3.plot(val.index,   val[TARGET],   color='#ffbe0b', linewidth=0.8, label='Val')
ax3.plot(test.index,  test[TARGET],  color='#f72585', linewidth=0.8, label='Test')
ax3.set_title('Train / Val / Test split', fontsize=9)
ax3.legend(fontsize=7)
ax3.set_ylabel('PAC (mg/L)')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=30, labelsize=7)

# Plot 3 — Feature correlations với target
ax4 = fig.add_subplot(gs[1, :2])
corr_with_target = df_fe[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
colors_c = ['#f72585' if v < 0 else '#00b4d8' for v in corr_with_target.values]
ax4.barh(corr_with_target.index, corr_with_target.values, color=colors_c, edgecolor='none')
ax4.axvline(x=0, color='white', linewidth=0.8)
ax4.set_title('Tương quan features với Target (PAC concentration)', fontsize=9)
ax4.set_xlabel('Pearson r')
ax4.grid(True, alpha=0.3, axis='x')
ax4.tick_params(axis='y', labelsize=8)

# Plot 4 — Lag autocorrelation của target
ax5 = fig.add_subplot(gs[1, 2])
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_fe[TARGET], ax=ax5, color='#7209b7')
ax5.set_xlim(0, 60)
ax5.set_title('Autocorrelation — PAC (lag 0-60 ngày)', fontsize=9)
ax5.set_xlabel('Lag (ngày)')
ax5.grid(True, alpha=0.3)

# Plot 5 — Seasonal pattern of target
ax6 = fig.add_subplot(gs[2, 0])
monthly = df_fe.groupby(df_fe.index.month)[TARGET].agg(['mean', 'std'])
months = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
ax6.fill_between(range(1,13),
                 monthly['mean'] - monthly['std'],
                 monthly['mean'] + monthly['std'], alpha=0.2, color='#3a86ff')
ax6.plot(range(1,13), monthly['mean'], 'o-', color='#3a86ff', linewidth=2, markersize=5)
ax6.set_xticks(range(1, 13))
ax6.set_xticklabels(months, fontsize=7)
ax6.set_title('Tính mùa vụ của PAC (trung bình theo tháng)', fontsize=9)
ax6.set_ylabel('PAC (mg/L)')
ax6.grid(True, alpha=0.3)

# Plot 6 — Scatter: Độ đục vs PAC
ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(df_fe['Doduc_vao_nha_may'], df_fe[TARGET],
            alpha=0.15, s=6, color='#ffbe0b')
ax7.set_xlabel('Độ đục đầu vào (NTU)', fontsize=8)
ax7.set_ylabel('PAC (mg/L)', fontsize=8)
ax7.set_title('Độ đục đầu vào vs PAC', fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 7 — Thể tích vs Nồng độ PAC scatter
ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(df_fe['Thetich_Jartest_PAC'], df_fe[TARGET],
            alpha=0.15, s=6, color='#fb5607')
ax8.set_xlabel('Thể tích Jartest PAC (mL)', fontsize=8)
ax8.set_ylabel('Nồng độ PAC (mg/L)', fontsize=8)
ax8.set_title('Thể tích vs Nồng độ PAC', fontsize=9)
ax8.grid(True, alpha=0.3)

plt.savefig('PREPROCESS_verification.png', bbox_inches='tight')
print("    Đã lưu: PREPROCESS_verification.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PREPROCESSING HOÀN THÀNH!")
print("=" * 62)
print(f"""
  Raw data     : 2191 rows × 50 cols
  After clean  : {len(df_fe)} rows × {df_fe.shape[1]} cols (gồm cả target)
  Features     : {len(all_feat_cols)} features
    - Raw features  : {len(FEATURE_COLS)} chỉ tiêu nước thô
    - Lag features  : {len(LAG_COLS)} cột × {len(LAGS)} lags = {len(LAG_COLS)*len(LAGS)}
    - Rolling stats : {len(ROLLING_COLS)} cột × 3 stats = {len(ROLLING_COLS)*3}
    - Temporal      : 8 (month, doy, week, sin/cos × 2, is_dry)
    - Log transform : 2 (Doduc_log, Doduc_log_lag1)
  
  Split (no data leakage):
    Train : {len(train)} rows  (2017-2020)
    Val   : {len(val)} rows  (2021)
    Test  : {len(test)} rows  (2022)
  
  Scaling: Min-Max [0, 1]  (fit on train only)
  
  Files saved in: dataset/
    train.csv / val.csv / test.csv
    processed_full.csv / scaler_params.json / metadata.json
""")
