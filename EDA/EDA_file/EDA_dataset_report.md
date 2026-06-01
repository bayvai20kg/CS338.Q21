# 📊 EDA Report — CLN_SG_V2.xlsx
## Dataset Chất lượng Nước Nhà máy Xử lý Nước Sài Gòn

---

## 1. Tổng quan Dataset

| Thuộc tính | Giá trị |
|---|---|
| **Tên file** | `CLN_SG_V2.xlsx` |
| **Số dòng (ngày đo)** | 2,191 |
| **Số cột (chỉ tiêu)** | 50 |
| **Khoảng thời gian** | 01/01/2017 → 31/12/2022 |
| **Tần số** | Daily (hàng ngày) |
| **Index chính** | Cột `Ngay` (datetime) |
| **Nguồn** | Nhà máy xử lý nước Sài Gòn (CLN = Chất Lượng Nước, SG = Sài Gòn) |

> [!NOTE]
> Dataset ghi lại hành trình xử lý nước theo **quy trình dây chuyền**: Sông Sài Gòn → Đầu vào nhà máy → Bể trộn → Bể lắng → Bể chứa → Đầu ra. Mỗi cột đo một chỉ tiêu chất lượng nước tại một điểm cụ thể trong quy trình.

---

## 2. Giải thích các cột theo nhóm

### 🧪 Nhóm pH — Độ axit/kiềm của nước

| Tên cột | Ý nghĩa | Vị trí đo | Missing |
|---|---|---|---|
| `pH_Song_SG` | pH sông Sài Gòn | Nguồn nước thô (intake) | 3.1% |
| `pH_vao_nha_may` | pH tại đầu vào nhà máy | Sau khi bơm từ sông vào | 0.2% |
| `pH_dau_be_tron` | pH tại đầu bể trộn | Sau khi thêm hóa chất coagulant | 0.2% |
| `pH_dau_be_lang` | pH tại đầu bể lắng | Đầu vào giai đoạn lắng | 0.5% |
| `pH_sau_be_lang` | pH sau bể lắng | Sau khi các hạt cặn đã lắng xuống | 0.2% |
| `pH_dau_be_chua_moi` | pH bể chứa mới | Bể chứa nước sạch mới | 0.2% |
| `pH_dau_be_chua_cu` | pH bể chứa cũ | Bể chứa nước sạch cũ | 0.2% |
| `pH_ra_nha_may` | pH tại đầu ra nhà máy | Nước thành phẩm xuất xưởng | 0.1% |

> **Tiêu chuẩn**: pH nước sinh hoạt: 6.5 – 8.5 (QCVN 01:2009/BYT)

---

### 💧 Nhóm Clo (Chlorine) — Chất khử trùng

| Tên cột | Ý nghĩa | Vị trí đo | Missing |
|---|---|---|---|
| `Clo_vao_nha_may` | Clo dư tại đầu vào | Nước thô vào nhà máy | 0.2% |
| `Clo_dau_be_tron` | Clo tại bể trộn | Sau bước pre-chlorination | 0.2% |
| `Clo_dau_be_chua_moi` | Clo bể chứa mới | Kiểm soát clo trong bể chứa | 0.1% |
| `Clo_dau_be_chua_cu` | Clo bể chứa cũ | Kiểm soát clo trong bể chứa cũ | 0.1% |
| `Clo_ra_nha_may` | Clo dư tại đầu ra | **Chỉ tiêu kiểm soát quan trọng nhất** | 0.1% |

> **Tiêu chuẩn**: Clo dư đầu ra: 0.2 – 1.0 mg/L

---

### 🌫️ Nhóm Độ đục (Turbidity)

| Tên cột | Ý nghĩa | Vị trí đo | Missing |
|---|---|---|---|
| `Doduc_vao_nha_may` | Độ đục nước thô (NTU) | Sông → Nhà máy | 0.2% |
| `Doduc_sau_lang` | Độ đục sau lắng (NTU) | Sau bể lắng | 0.2% |
| `Doduc_sau_loc` | Độ đục sau lọc (NTU) | Sau bể lọc cát | 0.2% |
| `Doduc_dau_be_chua_moi` | Độ đục bể chứa mới | Trong bể chứa | 0.1% |
| `Doduc_dau_be_chua_cu` | Độ đục bể chứa cũ | Trong bể chứa | 0.1% |
| `Doduc_ra_nha_may` | Độ đục đầu ra (NTU) | Nước thành phẩm | 0.1% |

> **Tiêu chuẩn**: Độ đục đầu ra ≤ 2 NTU (QCVN 01:2009/BYT)

---

### 🔩 Nhóm Kim loại nặng

| Tên cột | Ý nghĩa | Vị trí đo | Missing |
|---|---|---|---|
| `Mn_vao_nha_may` | Mangan (Mn) đầu vào (mg/L) | Nước thô | 0.2% |
| `Mn_sau_lang` | Mangan sau lắng (mg/L) | Sau bể lắng | 0.2% |
| `Mn_sau_loc` | Mangan sau lọc (mg/L) | Sau bể lọc | 0.2% |
| `Mn_ra_nha_may` | Mangan đầu ra (mg/L) | Nước thành phẩm | 0.1% |
| `Fe_vao_nha_may` | Sắt (Fe) đầu vào (mg/L) | Nước thô | 0.4% |
| `Fe_ra_nha_may` | Sắt (Fe) đầu ra (mg/L) | Nước thành phẩm | 0.2% |

> **Tiêu chuẩn**: Mn ≤ 0.1 mg/L; Fe ≤ 0.3 mg/L

---

### 🎨 Nhóm Màu sắc (Color) & Độ dẫn điện

| Tên cột | Ý nghĩa | Missing |
|---|---|---|
| `Mau_vao_nha_may_BK` | Màu nước thô (Pt-Co scale, đo quang học BK) | 0.3% |
| `Mau_vao_nha_may_thuc` | Màu nước thô (đo thực tế, quan trắc viên) | 0.3% |
| `Mau_ra_nha_may` | Màu nước đầu ra | 0.2% |
| `Man_song_saigon` | Độ mặn (TDS/Conductivity) nước sông SG (‰ hoặc mg/L) | 2.3% |
| `Man_ra_nha_may` | Độ mặn nước đầu ra | 0.2% |
| `Dodan_vao_nha_may` | Độ dẫn điện (µS/cm) đầu vào | 1.8% |
| `Dodan_ra_nha_may` | Độ dẫn điện (µS/cm) đầu ra | 1.7% |

---

### 🌬️ Nhóm Oxy hòa tan & Amoniac

| Tên cột | Ý nghĩa | Missing |
|---|---|---|
| `DO_vao_nha_may` | Oxy hòa tan đầu vào (mg/L) | 2.0% |
| `DO_ra_nha_may` | Oxy hòa tan đầu ra (mg/L) | 1.9% |
| `NH3_song_sai_gon` | Amoniac (NH₃) tại sông SG (mg/L) | 4.1% |
| `NH3_vao_nha_may` | Amoniac đầu vào nhà máy (mg/L) | 1.4% |
| `NH3_sau_loc` | Amoniac sau lọc (mg/L) | 6.6% |
| `NH3_ra_nha_may` | Amoniac đầu ra (mg/L) | 1.4% |

> **Tiêu chuẩn**: NH₃ ≤ 1.5 mg/L; DO trong nước sạch không bắt buộc nhưng là indicator chất lượng nguồn

---

### 🔬 Nhóm SS, Nhôm, Flo

| Tên cột | Ý nghĩa | Missing |
|---|---|---|
| `SS_vao_nha_may` | Chất rắn lơ lửng đầu vào (mg/L) | **7.4%** |
| `SS_sau_lang` | Chất rắn lơ lửng sau lắng (mg/L) | **7.4%** |
| `SS_ra_nha_may` | Chất rắn lơ lửng đầu ra (mg/L) | **7.3%** |
| `Al_vao_nha_may` | Nhôm đầu vào (mg/L) | **84%** ⚠️ |
| `Al_sau_loc` | Nhôm sau lọc (mg/L) | **97%** ⚠️ |
| `Al_ra_nha_may` | Nhôm đầu ra (mg/L) | **84%** ⚠️ |
| `F_dau_be_chua_moi` | Flo bể chứa mới (mg/L) | 0.2% |
| `F_ra_nha_may` | Flo đầu ra (mg/L) | 0.1% |

> [!WARNING]
> Các cột nhôm (`Al_*`) có **>84% missing** → không nên dùng làm feature chính. Cột `Al_sau_loc` gần như toàn bộ là NaN (97%).

---

### ⚗️ Nhóm Hóa chất xử lý (Jartest PAC)

| Tên cột | Ý nghĩa | Missing |
|---|---|---|
| `Thetich_Jartest_PAC` | Thể tích PAC dùng trong Jar test (mL) | 5.3% |
| `Nongdo_Jartest_PAC` | Nồng độ PAC tối ưu trong Jar test (mg/L) | 6.4% |
| `Nongdo_Jartest_Phen` | Nồng độ Phenol trong Jar test (mg/L) | **93%** ⚠️ |

> **PAC** = Poly Aluminium Chloride — chất keo tụ chính. Jar test là thí nghiệm lựa chọn liều lượng PAC tối ưu mỗi ngày.

---

## 3. Phân tích Missing Values

```
Nhóm ít missing (< 5%)  → Tin cậy cao: pH, Clo, Độ đục, Mn, Fe, Màu, F
Nhóm trung bình (5-10%) → Cần xử lý: SS, NH3, PAC, Man_song_saigon  
Nhóm nhiều (> 50%)      → Bỏ qua:    Al_*, Nongdo_Jartest_Phen
```

---

## 4. 🎯 Bài toán Time-Series Forecasting phù hợp

Dựa trên cấu trúc dataset và các paper trong folder `ref/` (LNN, LTC, LSTM, ARIMA), có 5 hướng bài toán:

---

### 🏆 Bài toán 1 (Khuyến nghị — phù hợp nhất với LNN/LTC)
**Dự báo liều lượng PAC tối ưu hàng ngày**

| | Chi tiết |
|---|---|
| **Target** | `Nongdo_Jartest_PAC` (nồng độ PAC tối ưu) |
| **Features** | `Doduc_vao_nha_may`, `pH_Song_SG`, `Man_song_saigon`, `NH3_song_sai_gon`, `Mn_vao_nha_may` |
| **Lý do** | PAC là biến quyết định chất lượng xử lý — dự báo trước giúp chuẩn bị hóa chất |
| **Khó khăn** | Tính phi tuyến cao, phụ thuộc nhiều yếu tố môi trường |
| **Model đề xuất** | **Liquid Neural Network (LTC)** ← đúng trọng tâm của project |

---

### 🥈 Bài toán 2
**Dự báo Độ đục đầu ra (`Doduc_ra_nha_may`)**

| | Chi tiết |
|---|---|
| **Target** | `Doduc_ra_nha_may` |
| **Features** | `Doduc_vao_nha_may`, `Nongdo_Jartest_PAC`, `pH_dau_be_tron`, `Mn_vao_nha_may` |
| **Lý do** | Chỉ tiêu pháp lý quan trọng nhất (≤ 2 NTU). Vi phạm → phải xả bỏ nước |
| **Model đề xuất** | LNN/LSTM, so sánh với ARIMA |

---

### 🥉 Bài toán 3
**Dự báo Mặn / NH3 tại sông Sài Gòn (nước thô)**

| | Chi tiết |
|---|---|
| **Target** | `Man_song_saigon` hoặc `NH3_song_sai_gon` |
| **Features** | Lag features của chính nó, `DO_vao_nha_may`, tháng trong năm |
| **Lý do** | Mặn và NH3 có tính mùa vụ rõ rệt → dự báo trước để điều chỉnh quy trình |
| **Model đề xuất** | LNN so với ARIMA, Seasonal LSTM |

---

### 🔍 Bài toán 4
**Phát hiện bất thường (Anomaly Detection)**

| | Chi tiết |
|---|---|
| **Target** | Tất cả chỉ tiêu đầu ra |
| **Lý do** | Phát hiện ngày có sự cố (pH đột biến, Mn vượt ngưỡng...) |
| **Model đề xuất** | Autoencoder + LNN, hoặc reconstruction error |

---

### 📊 Bài toán 5
**Multi-step Forecasting — Dự báo chuỗi pH đầu ra 7 ngày tới**

| | Chi tiết |
|---|---|
| **Target** | `pH_ra_nha_may` (t+1 đến t+7) |
| **Features** | `pH_Song_SG`, `pH_vao_nha_may`, `Clo_vao_nha_may`, `Doduc_vao_nha_may` |
| **Lý do** | Chuỗi pH liên tục, dữ liệu đầy đủ (~100%), phù hợp benchmark |
| **Model đề xuất** | LSTM vs LNN vs ARIMA (so sánh 3 phương pháp) |

---

## 5. Đặc điểm Time-Series của Dataset

| Đặc điểm | Kết quả |
|---|---|
| **Tính thời vụ** | ✅ Rõ ràng — Mặn cao T1-T4 (mùa khô), Độ đục cao T8-T11 (mùa mưa) |
| **Tự tương quan (ACF)** | ✅ Cao — Lag 1-30 ngày có ý nghĩa thống kê |
| **Xu hướng (Trend)** | ⚠️ Nhẹ — cần kiểm định ADF |
| **Dữ liệu liên tục** | ✅ Hàng ngày, không có khoảng trống thời gian lớn |
| **Phù hợp LNN/LTC** | ✅ Cao — Dataset có dynamics phức tạp, phi tuyến |

---

> [!TIP]
> **Gợi ý cho Final Project**: Sử dụng **Bài toán 1** (dự báo PAC) hoặc **Bài toán 2** (dự báo Độ đục đầu ra) để so sánh **LNN/LTC vs LSTM vs ARIMA**. Đây là hướng phù hợp nhất với các paper trong folder `ref/` và có thể tạo ra contribution rõ ràng.
