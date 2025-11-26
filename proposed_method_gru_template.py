#!/usr/bin/env python
# coding: utf-8
"""
Template cho Proposed Method với GRU thay thế LSTM
Giữ nguyên CEEMDAN + EWT preprocessing
"""

from PyEMD import CEEMDAN
import numpy
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, GRU  # Thay LSTM bằng GRU
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from math import sqrt
import pandas as pd

def create_dataset(dataset, look_back=1):
    """Convert an array of values into a dataset matrix"""
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def proposed_method_gru(new_data, i, look_back, data_partition, cap):
    """
    Proposed Method: CEEMDAN-EWT-GRU
    
    Parameters:
    -----------
    new_data : DataFrame
        Input data với columns: Month, Year, Date, P_avg (hoặc LV ActivePower)
    i : list
        List các tháng cần forecast (ví dụ: [1] cho tháng 1)
    look_back : int
        Số timesteps để look back (ví dụ: 6)
    data_partition : float
        Tỷ lệ train/test split (ví dụ: 0.8)
    cap : float
        Capacity của wind farm (để tính MAPE)
    
    Returns:
    --------
    dict : Dictionary chứa các metrics (MAPE, RMSE, MAE)
    """
    
    x = i
    data1 = new_data.loc[new_data['month'].isin(x)]  # Hoặc 'Month' tùy dataset
    data1 = data1.reset_index(drop=True)
    data1 = data1.dropna()
    
    # Lấy dữ liệu wind power
    datas = data1['LV ActivePower (kW)']  # Hoặc 'P_avg' cho France dataset
    datas_wind = pd.DataFrame(datas)
    dfs = datas
    s = dfs.values

    # ========== CEEMDAN DECOMPOSITION ==========
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    
    full_imf = pd.DataFrame(IMFs)
    ceemdan1 = full_imf.T
    
    # ========== EWT DENOISING ==========
    # Lấy IMF đầu tiên (tần số cao nhất) để denoise
    imf1 = ceemdan1.iloc[:, 0]
    imf_dataps = numpy.array(imf1)
    imf_datasetss = imf_dataps.reshape(-1, 1)
    imf_new_datasets = pd.DataFrame(imf_datasetss)

    import ewtpy
    # EWT với N=3 components
    ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)
    
    # Bỏ component thứ 3 (thường là noise)
    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1, skipna=True)
    
    # Kết hợp denoised IMF1 với các IMF còn lại
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)
    
    # ========== FORECASTING VỚI GRU ==========
    pred_test = []
    test_ori = []
    pred_train = []
    train_ori = []

    # Hyperparameters
    epoch = 100
    batch_size = 64
    neuron = 128  # Số units trong GRU layer
    lr = 0.001
    optimizer = 'Adam'

    # Forecast từng IMF/subseries
    for col in new_ceemdan:
        datasetss2 = pd.DataFrame(new_ceemdan[col])
        datasets = datasetss2.values
        
        # Train/Test split
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        # Create sequences
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train = pd.DataFrame(trainX)
        Y_train = pd.DataFrame(trainY)
        X_test = pd.DataFrame(testX)
        Y_test = pd.DataFrame(testY)
        
        # Standardization
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(X_train)
        y = sc_y.fit_transform(Y_train)
        X1 = sc_X.fit_transform(X_test)
        y1 = sc_y.fit_transform(Y_test)
        y = y.ravel()
        y1 = y1.ravel()
        
        # Reshape cho GRU: (samples, timesteps, features)
        trainX = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1], 1))

        # Set random seeds
        numpy.random.seed(1234)
        tf.random.set_seed(1234)
        
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # ========== GPU CONFIGURATION ==========
        # Sử dụng helper function để cấu hình GPU
        try:
            from gpu_config import configure_gpu
            gpu_config = configure_gpu(memory_growth=True)
        except ImportError:
            # Fallback nếu không có gpu_config.py
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU được phát hiện: {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    print(f"Lỗi cấu hình GPU: {e}")
            else:
                print("Không phát hiện GPU, sử dụng CPU")
        
        # ========== BUILD GRU MODEL ==========
        # Thay LSTM bằng GRU ở đây
        model = Sequential()
        model.add(GRU(units=neuron, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer)

        # Training với GPU (nếu có)
        model.fit(trainX, y, epochs=epoch, batch_size=batch_size, verbose=0)

        # Prediction
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test = numpy.array(y_pred_test).ravel()
        y_pred_test = pd.DataFrame(y_pred_test)
        y1 = pd.DataFrame(y1)
        y = pd.DataFrame(y)
        y_pred_train = numpy.array(y_pred_train).ravel()
        y_pred_train = pd.DataFrame(y_pred_train)

        # Inverse transform
        y_test = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)
        y_pred_test1 = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)

        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    # ========== AGGREGATE RESULTS ==========
    result_pred_test = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    # Tổng hợp kết quả từ tất cả các IMFs
    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    # ========== EVALUATION ==========
    dataframe = pd.DataFrame(dfs)
    dataset = dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train = pd.DataFrame(trainX)
    Y_train = pd.DataFrame(trainY)
    X_test = pd.DataFrame(testX)
    Y_test = pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X_train)
    y = sc_y.fit_transform(Y_train)
    X1 = sc_X.fit_transform(X_test)
    y1 = sc_y.fit_transform(Y_test)
    y = y.ravel()
    y1 = y1.ravel()

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1 = pd.DataFrame(y1)
    y = pd.DataFrame(y)

    y_test = sc_y.inverse_transform(y1)
    y_train = sc_y.inverse_transform(y)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    # Calculate metrics
    mape = numpy.mean((numpy.abs(y_test - a)) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)

    print('MAPE', mape.to_string())
    print('RMSE', rmse)
    print('MAE', mae)
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': a,
        'actual': y_test
    }


# ========== CÁC BIẾN THỂ KHÁC ==========

def proposed_method_bilstm(new_data, i, look_back, data_partition, cap):
    """
    Proposed Method với Bidirectional LSTM
    Giống proposed_method_gru nhưng thay GRU bằng Bidirectional(LSTM)
    """
    # ... (tương tự như trên)
    from keras.layers import Bidirectional, LSTM
    
    # Trong phần build model:
    model = Sequential()
    model.add(Bidirectional(LSTM(units=neuron), input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    # ... (phần còn lại giống)


def proposed_method_cnn_lstm(new_data, i, look_back, data_partition, cap):
    """
    Proposed Method với CNN-LSTM Hybrid
    """
    from keras.layers import Conv1D, MaxPooling1D, LSTM
    
    # Trong phần build model:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                     input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=neuron))
    model.add(Dense(1))
    # ... (phần còn lại giống)


def proposed_method_attention_lstm(new_data, i, look_back, data_partition, cap):
    """
    Proposed Method với Attention LSTM
    Cần implement attention mechanism
    """
    # Có thể sử dụng keras-self-attention package
    # hoặc implement custom attention layer
    pass


# ========== USAGE EXAMPLE ==========
"""
# Trong notebook:
from proposed_method_gru_template import proposed_method_gru

# Load data (giống như hiện tại)
import pandas as pd
df = pd.read_csv('dataset/final_la_haute_R0711.csv')
df['Date'] = pd.to_datetime(df['Date_time'])
df['Year'] = df['Date'].dt.year 
df['Month'] = df['Date'].dt.month 
new_data = df[['Month','Year','Date','P_avg']]
new_data = new_data[new_data.Year == 2017]
cap = max(new_data['P_avg'])

# Parameters
i = [1]  # January
look_back = 6
data_partition = 0.8

# Run model
results = proposed_method_gru(new_data, i, look_back, data_partition, cap)
"""

