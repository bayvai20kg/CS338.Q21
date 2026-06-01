# 🔧 Preprocessing Walkthrough — CLN_SG_V2
### Bài toán: Dự báo liều PAC tối ưu hàng ngày (`Nongdo_Jartest_PAC`)

---

## Kết quả tổng quan

| | Trước | Sau |
|---|---|---|
| **Số dòng** | 2,191 | 2,036 |
| **Số cột** | 50 | 59 (gồm target) |
| **Missing values** | Có | 0 |
| **Outliers** | Có | Clipped |
| **Features** | 11 raw | 58 engineered |
| **Scaling** | — | Min-Max [0, 1] |

---

## Biểu đồ xác minh

![Preprocessing Verification — phân phối target, train/val/test split, tương quan features](C:\Users\GIA KHANG\.gemini\antigravity\brain\6f06a85c-c754-4b1f-b858-e01bfd9130db\PREPROCESS_verification.png)

---

## Pipeline 8 bước

### Bước 1 — Load & Chuẩn hóa tên cột

```python
df = pd.read_excel('dataset/CLN_SG_V2.xlsx')
df.columns = df.columns.str.strip()   # xóa trailing spaces (vd: 'pH_vao_nha_may ')
df['Ngay'] = pd.to_datetime(df['Ngay'])
df = df.sort_values('Ngay').set_index('Ngay')
```

> [!NOTE]
> Một số cột có tên bị dư dấu cách (trailing whitespace) trong file Excel gốc — bước `.str.strip()` là bắt buộc.

---

### Bước 2 — Loại bỏ cột quá nhiều Missing (> 50%)

| Cột bị loại | Missing |
|---|---|
| `Al_vao_nha_may` | 84.1% |
| `Al_sau_loc` | 97.3% |
| `Al_ra_nha_may` | 84.1% |
| `Nongdo_Jartest_Phen` | 93.2% |

```python
missing_pct = df.isnull().sum() / len(df) * 100
DROP_COLS = missing_pct[missing_pct > 50].index.tolist()
df = df.drop(columns=DROP_COLS)   # 50 → 46 cột
```

---

### Bước 3 — Chọn Feature Set (11 chỉ tiêu đầu vào)

Chỉ dùng **chỉ tiêu tại đầu vào / nước thô** — tránh data leakage từ các bước xử lý sau:

| Feature | Tên đầy đủ | Missing ban đầu |
|---|---|---|
| `Doduc_vao_nha_may` | Độ đục đầu vào (NTU) | 0.2% |
| `pH_Song_SG` | pH sông Sài Gòn | 3.1% |
| `pH_vao_nha_may` | pH tại đầu vào nhà máy | 0.2% |
| `Man_song_saigon` | Độ mặn sông SG (mg/L) | 2.3% |
| `NH3_song_sai_gon` | Amoniac nước thô (mg/L) | 4.1% |
| `Mn_vao_nha_may` | Mangan đầu vào (mg/L) | 0.2% |
| `Fe_vao_nha_may` | Sắt đầu vào (mg/L) | 0.4% |
| `DO_vao_nha_may` | Oxy hòa tan đầu vào (mg/L) | 2.0% |
| `Mau_vao_nha_may_BK` | Màu nước thô (Pt-Co) | 0.3% |
| `SS_vao_nha_may` | Chất rắn lơ lửng đầu vào (mg/L) | 7.4% |
| `Thetich_Jartest_PAC` | Thể tích PAC trong Jar test (mL) | 5.3% |

> [!IMPORTANT]
> **Không dùng** `Doduc_sau_lang`, `pH_ra_nha_may`, hay các chỉ tiêu đầu ra khác làm feature — đây là kết quả của quá trình xử lý, không có trước thời điểm ra quyết định.

---

### Bước 4 — Xử lý Missing Values

```python
# 4a — Drop các ngày không có giá trị target (không impute target)
df = df.dropna(subset=['Nongdo_Jartest_PAC'])   # 2191 → 2050 rows

# 4b — Interpolate linear cho features (phù hợp chuỗi thời gian)
df[FEATURE_COLS] = (
    df[FEATURE_COLS]
    .interpolate(method='linear', limit_direction='both')
    .ffill().bfill()
)
# Kết quả: 0 missing values còn lại
```

**Lý do dùng linear interpolation thay vì mean/median fill:**
- Dữ liệu thời gian liên tục — giá trị giữa 2 ngày đo là xấp xỉ tuyến tính hợp lý
- Mean fill sẽ xóa đi tính mùa vụ cục bộ

---

### Bước 5 — Xử lý Outliers (IQR × 3 Clipping)

```python
for col in FEATURE_COLS:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)
```

| Cột | Outliers bị clip | Khoảng sau clip |
|---|---|---|
| `Doduc_vao_nha_may` | 21 | [-32.2, 105.8] NTU |
| `Man_song_saigon` | 14 | [-44.6, 133.6] mg/L |
| `Mau_vao_nha_may_BK` | 33 | [-205.2, 724.5] Pt-Co |
| `Thetich_Jartest_PAC` | 30 | [3.8, 41.6] mL |
| `pH_Song_SG` | 3 | [4.66, 7.79] |
| `NH3_song_sai_gon` | 6 | [-0.92, 2.36] mg/L |
| `Mn_vao_nha_may` | 6 | [-20.3, 294.7] mg/L |
| `SS_vao_nha_may` | 18 | [-24.8, 89.0] mg/L |

> [!NOTE]
> Dùng IQR × **3** (thay vì × 1.5 thông thường) để giữ lại các sự kiện đặc biệt (lũ, triều cường) mà vẫn loại bỏ lỗi đo lường.

---

### Bước 6 — Feature Engineering (58 features)

````carousel
#### Nhóm 1: Lag Features (25 features)

Tạo giá trị quá khứ cho 5 cột quan trọng nhất:

| Cột gốc | Lag được tạo |
|---|---|
| `Doduc_vao_nha_may` | lag 1, 2, 3, 7, 14 ngày |
| `Man_song_saigon` | lag 1, 2, 3, 7, 14 ngày |
| `pH_Song_SG` | lag 1, 2, 3, 7, 14 ngày |
| `Mn_vao_nha_may` | lag 1, 2, 3, 7, 14 ngày |
| `Nongdo_Jartest_PAC` *(target)* | lag 1, 2, 3, 7, 14 ngày |

```python
LAGS = [1, 2, 3, 7, 14]
for col in LAG_COLS:
    for lag in LAGS:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
```

> Lag của **target** (autoregressive features) rất quan trọng cho LNN/LTC — mô hình học được quán tính của hệ thống xử lý nước.

<!-- slide -->
#### Nhóm 2: Rolling Statistics (12 features)

Thống kê cửa sổ trượt 7 ngày và 14 ngày (shift 1 để tránh leakage):

| Cột gốc | Rolling features |
|---|---|
| `Doduc_vao_nha_may` | 7d_mean, 7d_std, 14d_mean |
| `Man_song_saigon` | 7d_mean, 7d_std, 14d_mean |
| `Mn_vao_nha_may` | 7d_mean, 7d_std, 14d_mean |
| `Nongdo_Jartest_PAC` | 7d_mean, 7d_std, 14d_mean |

```python
for col in ROLLING_COLS:
    df[f'{col}_roll7_mean']  = df[col].shift(1).rolling(7,  min_periods=1).mean()
    df[f'{col}_roll7_std']   = df[col].shift(1).rolling(7,  min_periods=1).std().fillna(0)
    df[f'{col}_roll14_mean'] = df[col].shift(1).rolling(14, min_periods=1).mean()
```

<!-- slide -->
#### Nhóm 3: Temporal Features (8 features)

```python
df['month']         = df.index.month           # 1-12
df['day_of_year']   = df.index.dayofyear       # 1-365
df['week']          = df.index.isocalendar().week

# Fourier encoding — tránh discontinuity tháng 12→1
df['sin_month']     = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month']     = np.cos(2 * np.pi * df['month'] / 12)
df['sin_doy']       = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cos_doy']       = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Đặc trưng địa lý miền Nam VN
df['is_dry_season'] = df['month'].isin([11,12,1,2,3,4]).astype(int)
```

> **Fourier encoding** cho tháng/ngày trong năm giúp model hiểu tính tuần hoàn mà không cần dùng embedding.

<!-- slide -->
#### Nhóm 4: Log Transform (2 features)

Độ đục (`Doduc_vao_nha_may`) có phân phối **lệch phải mạnh** (right-skewed) — log transform giúp normalize:

```python
df['Doduc_log']      = np.log1p(df['Doduc_vao_nha_may'])
df['Doduc_log_lag1'] = df['Doduc_log'].shift(1)
```

`log1p(x) = log(1 + x)` — xử lý an toàn khi x = 0.
````

---

### Bước 7 — Train / Val / Test Split (Time-Based)

```
Timeline:  2017 ──────── 2018 ──────── 2019 ──────── 2020 ║ 2021 ║ 2022
                         TRAIN (66.7%)                     VAL   TEST
                         1,359 rows                        358   319
```

> [!IMPORTANT]
> **Không dùng random split** cho time-series! Random split gây **data leakage** vì model sẽ thấy dữ liệu tương lai trong lúc train. Luôn split theo thứ tự thời gian.

---

### Bước 8 — Min-Max Normalization

```python
# Fit ONLY trên train — tránh look-ahead bias
for col in ALL_COLS:
    col_min = train[col].min()
    col_max = train[col].max()
    scaler_params[col] = {'min': col_min, 'max': col_max}

# Transform tất cả splits
def minmax_scale(df, params):
    for col in params:
        mn, mx = params[col]['min'], params[col]['max']
        df[col] = (df[col] - mn) / (mx - mn)
    return df

train_scaled = minmax_scale(train, scaler_params)
val_scaled   = minmax_scale(val,   scaler_params)   # dùng params của train
test_scaled  = minmax_scale(test,  scaler_params)   # dùng params của train
```

**Inverse transform khi đánh giá model:**
```python
import json
with open('dataset/scaler_params.json') as f:
    params = json.load(f)

def inverse_transform(y_scaled, col='Nongdo_Jartest_PAC'):
    mn = params[col]['min']
    mx = params[col]['max']
    return y_scaled * (mx - mn) + mn

# Ví dụ: tính MAE theo đơn vị mg/L gốc
y_pred_real = inverse_transform(y_pred_scaled)
y_true_real = inverse_transform(y_true_scaled)
mae = np.mean(np.abs(y_pred_real - y_true_real))
print(f"MAE = {mae:.3f} mg/L")
```

---

## Output Files

| File | Kích thước | Mô tả |
|---|---|---|
| `dataset/train.csv` | 1359 × 59 | Dữ liệu train (scaled) |
| `dataset/val.csv` | 358 × 59 | Dữ liệu validation (scaled) |
| `dataset/test.csv` | 319 × 59 | Dữ liệu test (scaled) |
| `dataset/processed_full.csv` | 2036 × 59 | Full data chưa scale |
| `dataset/scaler_params.json` | 59 entries | Min/max để inverse transform |
| `dataset/metadata.json` | — | Cấu hình đầy đủ pipeline |

---

## Cách dùng trong Model Training

```python
import pandas as pd, json

train = pd.read_csv('dataset/train.csv', index_col='Ngay', parse_dates=True)
val   = pd.read_csv('dataset/val.csv',   index_col='Ngay', parse_dates=True)
test  = pd.read_csv('dataset/test.csv',  index_col='Ngay', parse_dates=True)

with open('dataset/metadata.json') as f:
    meta = json.load(f)

TARGET       = meta['target']                  # 'Nongdo_Jartest_PAC'
FEATURE_COLS = meta['feature_cols']            # list 58 features

X_train = train[FEATURE_COLS].values          # (1359, 58)
y_train = train[TARGET].values                # (1359,)

X_val   = val[FEATURE_COLS].values
y_val   = val[TARGET].values

X_test  = test[FEATURE_COLS].values
y_test  = test[TARGET].values

# Reshape cho sequence models (LSTM / LNN):
# X shape: (samples, timesteps, features)  — timesteps thường = 7 hoặc 14
```

> [!TIP]
> Với **Liquid Neural Network (LTC)**, input shape là `(batch, timesteps, features)`. Nên thử `timesteps = 7` (1 tuần) hoặc `timesteps = 14` (2 tuần) — phù hợp với lag features đã tạo.
