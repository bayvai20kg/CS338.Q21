# ============================================================
# EDA - Dataset Chất Lượng Nước Nhà Máy Sài Gòn (CLN_SG_V2)
# Time-Series Exploratory Data Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import warnings
warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'axes.labelcolor':  '#c8cce8',
    'xtick.color':      '#8890c8',
    'ytick.color':      '#8890c8',
    'text.color':       '#e8eaf6',
    'grid.color':       '#2a2d4a',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  '#a0c4ff',
    'figure.dpi':       120,
    'savefig.dpi':      150,
    'savefig.facecolor': '#0f1117',
    'legend.facecolor': '#1a1d2e',
    'legend.edgecolor': '#3a3d5c',
    'legend.labelcolor': '#c8cce8',
})

PALETTE = ['#00b4d8', '#f72585', '#7209b7', '#3a86ff', '#ff006e',
           '#8338ec', '#06d6a0', '#ffbe0b', '#fb5607', '#4cc9f0']
ACCENT  = '#a0c4ff'

# ─── Load Data ────────────────────────────────────────────────────────────────
DATASET_PATH = r'dataset\CLN_SG_V2.xlsx'
df = pd.read_excel(DATASET_PATH)
df['Ngay'] = pd.to_datetime(df['Ngay'])
df = df.sort_values('Ngay').reset_index(drop=True)
df = df.set_index('Ngay')

print("=" * 60)
print("  EDA: CLN_SG_V2 - Chất Lượng Nước Nhà Máy Sài Gòn")
print("=" * 60)
print(f"\n  Số dòng (ngày đo):  {len(df):,}")
print(f"  Số cột (chỉ tiêu):  {df.shape[1]}")
print(f"  Khoảng thời gian:   {df.index.min().date()} → {df.index.max().date()}")
print(f"  Tổng ngày span:     {(df.index.max() - df.index.min()).days + 1} ngày")

# ─── Nhóm các cột theo ý nghĩa ────────────────────────────────────────────────
GROUPS = {
    'pH': [c for c in df.columns if 'pH' in c],
    'Clo (Chlorine)': [c for c in df.columns if 'Clo' in c],
    'Độ đục (Turbidity)': [c for c in df.columns if 'Doduc' in c],
    'Mangan (Mn)': [c for c in df.columns if 'Mn_' in c],
    'Sắt (Fe)': [c for c in df.columns if 'Fe_' in c],
    'Màu sắc (Color)': [c for c in df.columns if 'Mau' in c],
    'Mặn - Độ dẫn': [c for c in df.columns if 'Man' in c or 'Dodan' in c],
    'Oxy hòa tan (DO)': [c for c in df.columns if 'DO_' in c],
    'Amoniac (NH3)': [c for c in df.columns if 'NH3' in c],
    'Nhôm (Al)': [c for c in df.columns if 'Al_' in c],
    'Flo (F)': [c for c in df.columns if 'F_' in c],
    'Hóa chất xử lý': [c for c in df.columns if 'Jartest' in c or 'SS_' in c],
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Tổng quan Dataset
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
fig.suptitle('📊 Tổng quan Dataset CLN_SG_V2 — Chất lượng nước Nhà Máy Sài Gòn (2017–2022)',
             fontsize=15, fontweight='bold', color='#e8eaf6', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1A – Missing value heatmap
ax1 = fig.add_subplot(gs[0, :2])
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
colors_bar = ['#f72585' if v > 50 else '#ffbe0b' if v > 10 else '#06d6a0'
              for v in missing_pct.values]
bars = ax1.barh(missing_pct.index, missing_pct.values, color=colors_bar, edgecolor='none', height=0.7)
ax1.set_xlabel('% Giá trị bị thiếu', color=ACCENT)
ax1.set_title('Tỷ lệ Missing Values theo từng cột', pad=8)
ax1.axvline(x=50, color='#f72585', linestyle='--', linewidth=1.2, alpha=0.6, label='50% threshold')
ax1.axvline(x=10, color='#ffbe0b', linestyle='--', linewidth=1.2, alpha=0.6, label='10% threshold')
ax1.legend(fontsize=8)
ax1.set_xlim(0, 105)
for bar, val in zip(bars, missing_pct.values):
    if val > 1:
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=7.5, color='#e8eaf6')
ax1.invert_yaxis()
ax1.tick_params(axis='y', labelsize=7)

# 1B – Số quan sát theo năm
ax2 = fig.add_subplot(gs[0, 2])
obs_per_year = df.resample('Y').count().mean(axis=1)
years = obs_per_year.index.year
ax2.bar(years, obs_per_year.values, color=PALETTE[:len(years)], edgecolor='none', width=0.6)
ax2.set_title('Số quan sát trung bình / năm', pad=8)
ax2.set_xlabel('Năm')
ax2.set_ylabel('Số ngày đo')
for i, (yr, val) in enumerate(zip(years, obs_per_year.values)):
    ax2.text(yr, val + 2, f'{val:.0f}', ha='center', fontsize=9, color='#e8eaf6')

# 1C – Correlation heatmap (top 15 có ít missing)
ax3 = fig.add_subplot(gs[1, :])
good_cols = missing_pct[missing_pct < 10].index.tolist()
corr = df[good_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax3,
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor='#0f1117',
            annot=True, fmt='.2f', annot_kws={'size': 6.5},
            cbar_kws={'shrink': 0.6})
ax3.set_title('Ma trận tương quan — Các cột có dữ liệu đầy đủ (missing < 10%)', pad=8)
ax3.tick_params(axis='x', rotation=45, labelsize=7)
ax3.tick_params(axis='y', rotation=0,  labelsize=7)

plt.savefig('EDA_01_overview.png', bbox_inches='tight')
print("\n[✓] Đã lưu: EDA_01_overview.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Time-Series pH theo toàn bộ nhà máy
# ═══════════════════════════════════════════════════════════════════════════════
ph_cols = GROUPS['pH']
fig, axes = plt.subplots(len(ph_cols), 1, figsize=(18, 2.2 * len(ph_cols)), sharex=True)
fig.suptitle('🧪 Diễn biến pH theo thời gian — Toàn bộ quy trình xử lý (2017–2022)',
             fontsize=14, fontweight='bold', color='#e8eaf6')

for i, (col, ax) in enumerate(zip(ph_cols, axes)):
    weekly = df[col].resample('W').mean()
    ax.fill_between(weekly.index, weekly.values, alpha=0.25, color=PALETTE[i])
    ax.plot(weekly.index, weekly.values, color=PALETTE[i], linewidth=1.2, label=col)
    ax.axhline(y=7.5, color='#ffbe0b', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=6.5, color='#ffbe0b', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_ylabel('pH', fontsize=8)
    ax.set_ylim(5.5, 9.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.4)
    missing_n = df[col].isnull().sum()
    ax.set_title(f'{col}  (missing: {missing_n})', pad=3, fontsize=9, color=ACCENT)

axes[-1].xaxis.set_major_locator(MonthLocator(interval=3))
axes[-1].xaxis.set_major_formatter(DateFormatter('%m/%Y'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('EDA_02_pH_timeseries.png', bbox_inches='tight')
print("[✓] Đã lưu: EDA_02_pH_timeseries.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Độ đục, Mangan, Sắt — Chỉ tiêu quan trọng tại đầu vào
# ═══════════════════════════════════════════════════════════════════════════════
key_input_cols = {
    'Độ đục đầu vào (NTU)': 'Doduc_vao_nha_may',
    'Độ đục sau lắng (NTU)': 'Doduc_sau_lang',
    'Độ đục sau lọc (NTU)': 'Doduc_sau_loc',
    'Độ đục đầu ra (NTU)':  'Doduc_ra_nha_may',
    'Mangan đầu vào (mg/L)': 'Mn_vao_nha_may',
    'Mangan đầu ra (mg/L)':  'Mn_ra_nha_may',
    'Sắt đầu vào (mg/L)':   'Fe_vao_nha_may',
    'Sắt đầu ra (mg/L)':    'Fe_ra_nha_may',
}

fig, axes = plt.subplots(4, 2, figsize=(18, 14), sharex=True)
fig.suptitle('🏭 Diễn biến Độ đục / Mangan / Sắt theo quy trình xử lý (2017–2022)',
             fontsize=14, fontweight='bold', color='#e8eaf6')
axes = axes.flatten()

for i, (label, col) in enumerate(key_input_cols.items()):
    ax = axes[i]
    weekly = df[col].resample('W').mean()
    ax.fill_between(weekly.index, weekly.values, alpha=0.2, color=PALETTE[i])
    ax.plot(weekly.index, weekly.values, color=PALETTE[i], linewidth=1.2)
    ax.set_title(label, pad=5, fontsize=9, color=ACCENT)
    ax.grid(True, alpha=0.4)
    ax.set_ylabel(label.split('(')[-1].replace(')', ''), fontsize=8)

for ax in axes[-2:]:
    ax.xaxis.set_major_locator(MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(DateFormatter('%m/%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('EDA_03_turbidity_mn_fe.png', bbox_inches='tight')
print("[✓] Đã lưu: EDA_03_turbidity_mn_fe.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Phân phối thống kê (boxplot theo năm)
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_COLS = ['pH_Song_SG', 'Doduc_vao_nha_may', 'Mn_vao_nha_may', 'Fe_vao_nha_may',
               'Man_song_saigon', 'NH3_song_sai_gon', 'DO_vao_nha_may', 'Clo_vao_nha_may']
TARGET_LABELS = ['pH Sông SG', 'Độ đục đầu vào', 'Mangan đầu vào',
                 'Sắt đầu vào', 'Mặn sông SG', 'NH3 sông SG', 'DO đầu vào', 'Clo đầu vào']

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('📦 Phân phối các chỉ tiêu đầu vào theo năm — Boxplot',
             fontsize=14, fontweight='bold', color='#e8eaf6')
axes = axes.flatten()

df_plot = df.copy()
df_plot['Year'] = df_plot.index.year

for i, (col, label) in enumerate(zip(TARGET_COLS, TARGET_LABELS)):
    ax = axes[i]
    year_groups = [df_plot[df_plot['Year'] == yr][col].dropna() for yr in range(2017, 2023)]
    bp = ax.boxplot(year_groups, patch_artist=True, notch=False,
                    medianprops=dict(color='#ffbe0b', linewidth=2))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#f72585', alpha=0.4, markersize=3)
    ax.set_xticklabels(range(2017, 2023), fontsize=8)
    ax.set_title(label, pad=5, fontsize=9, color=ACCENT)
    ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('EDA_04_boxplot_by_year.png', bbox_inches='tight')
print("[✓] Đã lưu: EDA_04_boxplot_by_year.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Tính thời vụ — monthly pattern
# ═══════════════════════════════════════════════════════════════════════════════
SEASONAL_COLS = ['pH_Song_SG', 'Doduc_vao_nha_may', 'Man_song_saigon',
                 'NH3_song_sai_gon', 'Mn_vao_nha_may', 'DO_vao_nha_may']
SEASONAL_LABELS = ['pH Sông SG', 'Độ đục đầu vào (NTU)', 'Mặn sông SG (‰)',
                   'NH3 sông SG (mg/L)', 'Mangan đầu vào (mg/L)', 'DO đầu vào (mg/L)']

df_monthly = df.copy()
df_monthly['Month'] = df_monthly.index.month
month_names = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('🌊 Tính thời vụ (Seasonality) — Trung bình theo tháng (2017–2022)',
             fontsize=14, fontweight='bold', color='#e8eaf6')
axes = axes.flatten()

for i, (col, label) in enumerate(zip(SEASONAL_COLS, SEASONAL_LABELS)):
    ax = axes[i]
    monthly_mean = df_monthly.groupby('Month')[col].mean()
    monthly_std  = df_monthly.groupby('Month')[col].std()
    x = monthly_mean.index
    ax.fill_between(x, monthly_mean - monthly_std, monthly_mean + monthly_std,
                    alpha=0.2, color=PALETTE[i], label='±1 std')
    ax.plot(x, monthly_mean.values, 'o-', color=PALETTE[i], linewidth=2, markersize=6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=8)
    ax.set_title(label, pad=5, fontsize=9, color=ACCENT)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('EDA_05_seasonality.png', bbox_inches='tight')
print("[✓] Đã lưu: EDA_05_seasonality.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: PAC Jartest — Lượng hóa chất xử lý
# ═══════════════════════════════════════════════════════════════════════════════
pac_cols = ['Thetich_Jartest_PAC', 'Nongdo_Jartest_PAC']
pac_labels = ['Thể tích Jartest PAC (mL)', 'Nồng độ Jartest PAC (mg/L)']

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
fig.suptitle('⚗️  Hóa chất xử lý PAC — Diễn biến & tương quan với Độ đục đầu vào',
             fontsize=13, fontweight='bold', color='#e8eaf6')

for i, (col, label) in enumerate(zip(pac_cols, pac_labels)):
    ax = axes[i]
    weekly = df[col].resample('W').mean()
    ax.fill_between(weekly.index, weekly.values, alpha=0.2, color=PALETTE[i])
    ax.plot(weekly.index, weekly.values, color=PALETTE[i], linewidth=1.2, label=label)
    ax.set_title(label, pad=4, fontsize=9, color=ACCENT)
    ax.grid(True, alpha=0.4)

axes[-1].xaxis.set_major_locator(MonthLocator(interval=3))
axes[-1].xaxis.set_major_formatter(DateFormatter('%m/%Y'))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('EDA_06_PAC_chemical.png', bbox_inches='tight')
print("[✓] Đã lưu: EDA_06_PAC_chemical.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Autocorrelation — kiểm tra tính time-series
# ═══════════════════════════════════════════════════════════════════════════════
from pandas.plotting import autocorrelation_plot

ACF_COLS = ['pH_Song_SG', 'Doduc_vao_nha_may', 'Mn_vao_nha_may', 'Man_song_saigon']
ACF_LABELS = ['pH Sông SG', 'Độ đục đầu vào', 'Mangan đầu vào', 'Mặn sông SG']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('📈 Autocorrelation — Tính chất chuỗi thời gian của các chỉ tiêu quan trọng',
             fontsize=13, fontweight='bold', color='#e8eaf6')
axes = axes.flatten()

for i, (col, label) in enumerate(zip(ACF_COLS, ACF_LABELS)):
    ax = axes[i]
    series = df[col].dropna()
    autocorrelation_plot(series, ax=ax, color=PALETTE[i])
    ax.set_title(f'Autocorrelation: {label}', pad=5, fontsize=9, color=ACCENT)
    ax.set_xlabel('Lag (ngày)', fontsize=8)
    ax.set_ylabel('ACF', fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 365)

plt.tight_layout()
plt.savefig('EDA_07_autocorrelation.png', bbox_inches='tight')
print("[✓] Đã lưu: EDA_07_autocorrelation.png")
plt.close()

print("\n" + "=" * 60)
print("  ✅ EDA hoàn thành! Tất cả 7 biểu đồ đã được lưu.")
print("=" * 60)
print("""
📌 TÓM TẮT DATASET:
  • Tên file   : CLN_SG_V2.xlsx
  • Dữ liệu   : Nhà máy xử lý nước Sài Gòn
  • Thời gian  : 01/01/2017 – 31/12/2022 (6 năm, 2191 ngày)
  • Tần số     : Daily (hàng ngày)
  • Số cột     : 50 chỉ tiêu chất lượng nước
  • Index      : Cột 'Ngay' (datetime)

🎯 CÁC BÀI TOÁN TIME-SERIES CÓ THỂ THỰC HIỆN:
  1. Dự báo pH đầu ra từ các chỉ tiêu đầu vào
  2. Dự báo Độ đục sau xử lý (dùng Liquid Neural Network)
  3. Tối ưu lượng PAC cần dùng mỗi ngày
  4. Phát hiện dị thường (anomaly) trong chất lượng nước
  5. Dự báo Mặn / NH3 sông Sài Gòn theo mùa
""")
