# CEEMDAN-EWT-LSTM
## Wind Power Forecasting Based on Hybrid CEEMDAN- EWT Deep Learning Method

This is the original source code used for all experiments in the paper [Wind Power Forecasting Based on Hybrid CEEMDAN-EWT Deep Learning Method](https://www.sciencedirect.com/science/article/pii/S0960148123012727)

Please cite the paper if you utilize the code in this paper:


Karijadi, I., Chou, S. Y., & Dewabharata, A. (2023). Wind power forecasting based on hybrid CEEMDAN-EWT deep learning method. Renewable Energy, 119357.


## Authors

*Irene Karijadi*, Shuo-Yan Chou, Anindhita Dewabharata


**corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)*

## Background
A precise wind power forecast is required for the renewable energy platform to function effectively. By having a precise wind power forecast, the power system can better manage its supply and ensure grid reliability. However, the nature of wind power generation is intermittent and exhibits high randomness, which poses a challenge to obtain accurate forecasting results. In this study, a hybrid method is proposed based on Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN), Empirical Wavelet Transform (EWT), and deep learning-based Long Short-Term Memory (LSTM) for ultra-short-term wind power forecasting. A combination of CEEMDAN and EWT is used as the preprocessing technique, where CEEMDAN is first employed to decompose the original wind power data into several subseries and EWT denoising technique is used to denoise the highest frequency series generated from CEEMDAN. Then, LSTM is utilized to forecast all the subseries from CEEMDAN-EWT process, and the forecasting results of each subseries are aggregated to achieve the final forecasting results. The proposed method is validated on real-world wind power data in France and Turkey. Our experimental results demonstrate that the proposed method can forecast more accurately than the benchmarking methods.

## Framework
This is the framework of the proposed method      


![Proposed Method Framework](https://github.com/irenekarijadi/CEEMDAN-EWT-LSTM/assets/28720072/922f6554-ff1c-4acb-b8c0-2ef167fc0d51)


## Prerequisites
The proposed method is coded in Python 3.7.6 and the experiments were performed on Intel Core i3-8130U CPU, 2.20GHz, with a memory size of 4.00 GB.

### System Requirements
- Python 3.7.6 (hoặc 3.7.x)
- pip
- Jupyter Notebook hoặc JupyterLab
- GPU (optional, nhưng được khuyến nghị để tăng tốc độ training)

### Installation

#### 1. Tạo môi trường ảo Python 3.7

**Cách 1: Sử dụng venv (khuyến nghị)**
```bash
# Cài Python 3.7 nếu chưa có
# Ubuntu/Debian:
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.7 python3.7-venv python3.7-dev

# Tạo môi trường ảo
python3.7 -m venv venv37

# Kích hoạt môi trường
source venv37/bin/activate  # Linux/Mac
# hoặc
venv37\Scripts\activate  # Windows
```

**Cách 2: Sử dụng Conda**
```bash
conda create -n ceemdan-ewt-lstm python=3.7.6
conda activate ceemdan-ewt-lstm
```

#### 2. Cài đặt dependencies

```bash
# Nâng cấp pip
pip install --upgrade pip

# Cài đặt tất cả packages từ requirements.txt
pip install -r requirements.txt
```

#### 3. Đăng ký kernel cho Jupyter (nếu dùng venv)

```bash
# Kích hoạt môi trường ảo trước
source venv37/bin/activate

# Cài ipykernel
pip install ipykernel

# Đăng ký kernel
python -m ipykernel install --user --name venv37 --display-name "Python 3.7.6 (venv37)"
```

### Dependencies

Các package chính được liệt kê trong `requirements.txt`:
* EMD-signal==0.2.10
* pandas==0.25.3
* keras==2.4.3
* tensorflow>=2.0.0
* scikit-learn==0.22.1
* numpy==1.18.1
* matplotlib
* ewtpy==0.2
* PyEMD (được cài tự động qua EMD-signal)
* scipy (dependency của các package khác)
* bayesian-optimization (cho hyperparameter tuning)


## Dataset
The performance of the proposed method is tested by using wind power datasets in two different countries. The first dataset is from a wind farm with an installed capacity of 2050 kW located in France, and the second dataset is from a wind farm with an installed capacity of 3600 kW located in Turkey.

### Dataset Structure

**France Dataset** (`dataset/final_la_haute_R0711.csv`):
- Cột: `Date_time`, `P_avg`
- Đã được xử lý sẵn trong notebook

**Turkey Dataset** (`dataset/T1.csv`):
- Cột: `Date/Time`, `LV ActivePower (kW)`, `Wind Speed (m/s)`, ...
- Notebook tự động xử lý để tạo cột `month` từ `Date/Time`

## Cách Sử Dụng

### 1. Khởi động Jupyter Notebook

```bash
# Kích hoạt môi trường ảo
source venv37/bin/activate

# Khởi động Jupyter
jupyter notebook
# hoặc
jupyter lab
```

**Lưu ý**: Chọn kernel `venv37` (hoặc kernel tương ứng với môi trường ảo của bạn) trong Jupyter.

### 2. Chạy các Experiments

#### Notebook 1: France Dataset
```python
# File: 1. Experiments for France Dataset.ipynb
# Chạy tất cả cells để:
# - Load và preprocess dữ liệu France
# - Train và test các models: SVR, ANN, RF, LSTM, EMD-LSTM, EEMD-LSTM, CEEMDAN-LSTM
# - Chạy proposed method (CEEMDAN-EWT-LSTM)
# - Chạy enhanced version (CEEMDAN-EWT-GRU) - HÀM MỚI
```

#### Notebook 2: Turkey Dataset
```python
# File: 2. Experiments for Turkey Dataset.ipynb
# Tương tự notebook 1 nhưng cho dataset Turkey
# Notebook tự động xử lý Date/Time để tạo cột 'month'
```

#### Notebook 3 & 4: Time Series Cross Validation
```python
# File: 3. Experiments for France Dataset-Time Series Cross Validation.ipynb
# File: 4. Experiments for Turkey Dataset-Time Series Cross Validation.ipynb
# Thực hiện cross-validation với các folds khác nhau
```

#### Notebook 5: Comparative Experiments
```python
# File: 5. Comparative experiments.ipynb
# So sánh kết quả giữa các phương pháp
```

### 3. Các Hàm Chính

#### Hàm gốc (từ paper):
- `proposed_method()`: CEEMDAN-EWT-LSTM (phương pháp gốc)

#### Hàm mới (enhanced version):
- `ceemdan_ewt_evl()`: CEEMDAN-EWT-GRU
  - Giữ nguyên preprocessing: CEEMDAN + EWT
  - Thay LSTM bằng GRU (nhẹ hơn, nhanh hơn)
  - Tự động cấu hình GPU nếu có

#### Các hàm benchmark:
- `svr_model()`: Support Vector Regression
- `ann_model()`: Artificial Neural Network
- `rf_model()`: Random Forest
- `lstm_model()`: LSTM standalone
- `emd_lstm()`: EMD-LSTM
- `eemd_lstm()`: EEMD-LSTM
- `ceemdan_lstm()`: CEEMDAN-LSTM

### 4. Cấu Hình GPU (Optional)

Hàm `ceemdan_ewt_evl()` tự động phát hiện và sử dụng GPU nếu có. Để kiểm tra GPU:

```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

Để vô hiệu hóa GPU (chỉ dùng CPU):
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### 5. Tham Số Mặc Định

Các tham số có thể điều chỉnh trong notebook:

```python
i = [1]  # Tháng cần forecast (1 = January, 2 = February, ...)
look_back = 6  # Số timesteps để look back
data_partition = 0.8  # Tỷ lệ train/test split (0.8 = 80% train, 20% test)
```

### 6. Kết Quả

Mỗi hàm sẽ in ra 3 metrics:
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error

## Experiments

Các notebook thực nghiệm:

* `1. Experiments for France Dataset.ipynb` - Train và test các methods trên France dataset
* `2. Experiments for Turkey Dataset.ipynb` - Train và test các methods trên Turkey dataset
* `3. Experiments for France Dataset-Time Series Cross Validation.ipynb` - Cross-validation cho France dataset
* `4. Experiments for Turkey Dataset-Time Series Cross Validation.ipynb` - Cross-validation cho Turkey dataset
* `5. Comparative experiments.ipynb` - So sánh kết quả giữa các phương pháp

## Ensemble Learning (PyTorch + LightGBM)

Ngoài các notebook, repo còn có script `ensemble_ceemdan_ewt_lightgbm.py` để chạy CEEMDAN + EWT → PyTorch LSTM → LightGBM residual trực tiếp trên GPU.

### Cài đặt phụ thuộc

```bash
python3 -m venv venv-el
source venv-el/bin/activate
pip install -r requirements_ensemble.txt
```

### Cách chạy cơ bản

```bash
python ensemble_ceemdan_ewt_lightgbm.py \
  --data-path dataset/final_la_haute_R0711.csv \
  --target-column P_avg \
  --datetime-column Date_time \
  --look-back 12 \
  --sequence-mode per-imf \
  --imf-bidirectional \
  --imf-epochs 120 \
  --residual-lags 5
```

Script tự sắp xếp dữ liệu theo `--datetime-column`, tạo train/test theo `--train-ratio` (mặc định 0.8), training LSTM trên GPU (nếu có CUDA) rồi cải thiện residual bằng LightGBM. Kết quả cuối in MAPE/RMSE/MAE và cả metric của phần residual.

### Lọc tháng hoặc cross-validation theo tháng

- `--month-column`: tên cột chứa tháng (mặc định `Month`). Nếu dataset chưa có, script sẽ tạo từ cột thời gian.
- `--months`: danh sách tháng cần huấn luyện/đánh giá. Ví dụ: `--months 1` hoặc `--months 1 2 3`.
- `--month-folds`: chạy liên tiếp nhiều nhóm tháng để mô phỏng cross-validation. Ví dụ:

```bash
python ensemble_ceemdan_ewt_lightgbm.py \
  --data-path dataset/T1.csv \
  --target-column "LV ActivePower (kW)" \
  --datetime-column "Date/Time" \
  --datetime-format "%d %m %Y %H:%M" \
  --dayfirst \
  --weather-columns "Wind Speed (m/s)" "Theoretical_Power_Curve (KWh)" "Wind Direction (°)" \
  --look-back 12 \
  --sequence-mode per-imf \
  --imf-hidden-size 192 \
  --imf-bidirectional \
  --train-ratio 0.9 \
  --months 1 2 3
```

Hoặc cross-val:

```bash
python ensemble_ceemdan_ewt_lightgbm.py \
  --data-path dataset/final_la_haute_R0711.csv \
  --target-column P_avg \
  --datetime-column Date_time \
  --sequence-mode per-imf \
  --month-folds "1" "2" "3,4" \
  --look-back 12
```

Mỗi fold sẽ in rõ label `Run: foldX_months_...` cùng các metric tương ứng.

### Các tham số Ensemble quan trọng

- `--sequence-mode`: `per-imf` (default) huấn luyện một model PyTorch riêng cho từng IMF (giống proposed_method), `shared` thì dùng một model đa biến duy nhất.
- `--imf-*`: cấu hình cho các IMF models (kiểu cell `gru`/`lstm`, số tầng, hidden size, dropout, epochs, lr, patience, batch size, `--imf-bidirectional`...).
- `--shared-*`: thông số cho mô hình `shared` (nếu sử dụng).
- `--residual-lags`, `--lgbm-estimators`, `--lgbm-learning-rate`: tinh chỉnh LightGBM residual để hoàn thiện ensemble.
- `--month-folds`: truyền nhiều nhóm tháng (ví dụ `"1,2,3"` `"4,5,6"`) để chạy cross-validation tương tự notebook 3/4.

Sau khi chạy, log sẽ hiển thị:
- Metrics của mô hình sequence (để so sánh trực tiếp với proposed_method).
- Metrics sau khi cộng residual từ LightGBM (MAPE/RMSE/MAE).
- Diagnostic riêng của residual model (RMSE/MAE trên phần residual).

## Cấu Trúc Dự Án

```
CEEMDAN-EWT-LSTM/
├── dataset/                          # Thư mục chứa dữ liệu
│   ├── final_la_haute_R0711.csv     # France dataset
│   └── T1.csv                        # Turkey dataset
├── myfunctions.py                    # Functions cho Turkey dataset
├── myfunctions_france.py             # Functions cho France dataset
├── gpu_config.py                    # Helper cho GPU configuration
├── proposed_method_gru_template.py  # Template cho GRU version
├── requirements.txt                  # Dependencies
├── README.md                         # File này
└── *.ipynb                           # Các notebook experiments
```

## Tính Năng Mới

### Enhanced Version: CEEMDAN-EWT-GRU

Hàm `ceemdan_ewt_evl()` là phiên bản cải tiến với:
- ✅ Giữ nguyên core: CEEMDAN + EWT preprocessing
- ✅ Thay LSTM bằng GRU (Gated Recurrent Unit)
- ✅ Tự động cấu hình GPU
- ✅ Tương thích với cả France và Turkey dataset

**So sánh với phương pháp gốc:**
- GRU nhẹ hơn LSTM (~30% ít parameters)
- Training nhanh hơn
- Hiệu suất tương đương hoặc tốt hơn trong nhiều trường hợp

## Troubleshooting

### Lỗi: "No module named 'xxx'"
```bash
# Đảm bảo đã kích hoạt môi trường ảo
source venv37/bin/activate

# Cài lại dependencies
pip install -r requirements.txt
```

### Lỗi: "KeyError: 'month'" (Turkey dataset)
- Notebook đã tự động xử lý, không cần làm gì thêm
- Nếu vẫn lỗi, kiểm tra cell 1 trong notebook Turkey đã chạy chưa

### GPU không được sử dụng
- Kiểm tra TensorFlow có phát hiện GPU: `tf.config.list_physical_devices('GPU')`
- Hàm `ceemdan_ewt_evl()` tự động sử dụng GPU nếu có
- Nếu muốn force CPU: `os.environ['CUDA_VISIBLE_DEVICES'] = ''`

### Lỗi version compatibility
- Đảm bảo Python 3.7.x
- Sử dụng đúng versions trong requirements.txt
- Không nâng cấp TensorFlow lên version quá mới (giữ >=2.0.0, <2.4)

## Citation

Nếu sử dụng code này, vui lòng cite paper gốc:

```
Karijadi, I., Chou, S. Y., & Dewabharata, A. (2023). Wind power forecasting based on hybrid CEEMDAN-EWT deep learning method. Renewable Energy, 119357.
```

## Authors

*Irene Karijadi*, Shuo-Yan Chou, Anindhita Dewabharata

**corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)**


