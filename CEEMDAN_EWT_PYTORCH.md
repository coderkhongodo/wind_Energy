## CEEMDAN + EWT + PyTorch Models

File chính: `ceemdan_ewt_models.py`

### 1. Môi trường & cài đặt

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_ceemdan_ewt_pytorch.txt
```

### 2. Chuẩn bị dữ liệu

- Tập France: đã có cột `Month`, có thể truyền danh sách tháng bằng `--months 1 2 3`.
- Tập Turkey: không có cột tháng, hãy cung cấp `--timestamp_col` để script tự sinh tháng hoặc bỏ qua `--months` để dùng toàn bộ dữ liệu.
- `--target_col` phải khớp với tên cột công suất (`P_avg` cho France, `LV ActivePower (kW)` cho Turkey).

### 3. Tham số chính

| Tham số | Ý nghĩa |
| --- | --- |
| `--model {tcn,informer,bilstm_attn}` | Chọn mô hình PyTorch chạy trên GPU nếu có. |
| `--look_back` | Độ dài chuỗi đầu vào cho mỗi mẫu. |
| `--train_ratio` | Tỉ lệ train/test cho từng thành phần CEEMDAN. |
| `--cap` | Công suất định mức để tính MAPE (nếu bỏ trống sẽ dùng max của dữ liệu test). |
| `--months` | Danh sách tháng cần train (có thể trống). |
| `--output_path` | Đường dẫn lưu CSV kết quả dự báo. |

### 4. Ví dụ chạy

Huấn luyện France với TCN:

```bash
python ceemdan_ewt_models.py \
  --data_path dataset/final_la_haute_R0711.csv \
  --target_col P_avg \
  --month_col Month \
  --months 1 2 3 4 \
  --model tcn \
  --look_back 48 \
  --train_ratio 0.8 \
  --epochs 60 \
  --batch_size 128 \
  --cap 2100 \
  --output_path outputs/france_tcn.csv
```

Huấn luyện Turkey với BiLSTM + Attention cho toàn bộ dữ liệu:

```bash
python ceemdan_ewt_models.py \
  --data_path dataset/wind_turkey.csv \
  --target_col "LV ActivePower (kW)" \
  --timestamp_col Date \
  --model bilstm_attn \
  --look_back 96 \
  --train_ratio 0.75 \
  --epochs 80 \
  --batch_size 64 \
  --cap 2000 \
  --output_path outputs/turkey_bilstm.csv
```

Chạy Informer-lite (transformer rút gọn):

```bash
python ceemdan_ewt_models.py \
  --data_path dataset/final_la_haute_R0711.csv \
  --target_col P_avg \
  --month_col Month \
  --months 5 6 7 8 \
  --model informer \
  --look_back 72 \
  --epochs 80 \
  --hidden_dim 192 \
  --output_path outputs/france_informer.csv
```

### 5. GPU

Script tự động chọn `cuda` khi có GPU khả dụng. Kiểm tra bằng:

```bash
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
PY
```

