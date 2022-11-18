import numpy as np
import pandas as pd
import keras
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from math import sqrt
from numpy import array
from utils.logger import Logger
from environment.env_config import STEP

save_model_path = "./model/lstm.h5"
# save_model_path = "../model/lstm.h5"
oai_delay_data_path = "./delay_data/dataset/oai_时延测试数据.xlsx"
# oai_delay_data_path = "./dataset/oai_时延测试数据.xlsx"
test_data = pd.read_excel(os.path.abspath(oai_delay_data_path))
test_oai_delay_data = test_data['时延/ms']
test_processed_data = list(test_oai_delay_data[test_oai_delay_data > 25].index)
test_processed_data = test_oai_delay_data.drop(test_processed_data, axis=0)
min_delay = test_processed_data.min()
max_delay = test_processed_data.max()


def train_lstm(data, test_flag, n_input, n_batch, n_epoch, n_neurons):
    train_data = data[:data.shape[0]-test_flag, :, :]
    train_x_data = train_data[:, :n_input, :]
    train_y_data = train_data[:, n_input:, :]
    test_data = data[data.shape[0]-test_flag:, :]
    test_x_data = test_data[:, :n_input, :]
    test_y_data = test_data[:, n_input:, :]
    print("train_data shape:", train_data.shape)
    print("test_data shape:", test_data.shape)
    print("train_x_data shape:", train_x_data.shape)
    print("test_x_data shape:", test_x_data.shape)
    print("train_y_data shape:", train_y_data.shape)
    print("test_y_data shape:", test_y_data.shape)
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(train_x_data.shape[1], train_x_data.shape[2])))
    model.add(LSTM(n_neurons))
    model.add(Dropout(0.2))
    model.add(Dense(train_y_data.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    print(model.summary())
    model.fit(train_x_data.numpy(), train_y_data.numpy(), epochs=n_epoch, batch_size=n_batch, validation_data=(test_x_data.numpy(), test_y_data.numpy()), verbose=1, shuffle=False)
    test_pre_data = model.predict(test_x_data.numpy())
    return model, train_x_data, train_y_data, test_x_data, test_y_data, test_pre_data

def save_lstm_model(model):
    model.save(os.path.abspath(save_model_path))
    print("successfully save!")

def load_lstm_model():
    if os.path.exists(os.path.abspath(save_model_path)):
        Logger.logger.info("------------------------------------ The LSTM is existed and can be loaded ------------------------------------ ")
        model = keras.models.load_model(os.path.abspath(save_model_path))
        return model
    else:
        Logger.logger.error("The LSTM is not existed and you must train and save the LSTM prediction model firstly by running lstm.py ")
        raise

def deal_data_supervised(n_input, n_output):
    data = pd.read_excel(os.path.abspath(oai_delay_data_path))
    oai_delay_data = data['时延/ms']
    processed_data = list(oai_delay_data[oai_delay_data>25].index)
    processed_data = oai_delay_data.drop(processed_data, axis=0)
    scaler_oai_data = MinMaxScaler(feature_range=(0, 1))
    scaler_data = scaler_oai_data.fit_transform(torch.tensor(processed_data.values).reshape(-1, 1))
    scaler_data = torch.from_numpy(scaler_data)
    supervised_data = torch.zeros((scaler_data.shape[0] - n_input, n_input + n_output))
    for i in range(supervised_data.shape[0]):
        supervised_data[i] = torch.Tensor([item for item in scaler_data[i:i+n_input+n_output]])
    supervised_data = supervised_data.reshape(supervised_data.shape[0], supervised_data.shape[1], 1)
    return supervised_data


def evaluate(model, test_x):
    # test_x.shape : n * n_lag * 1
    delay_pre = model.predict(test_x)
    delay_pre = delay_pre * (max_delay - min_delay) + min_delay
    return delay_pre[0][0]

def delay_predict(data, lstm_model):
    # data: list
    Logger.logger.debug("The last three float action delay datas are %lf ms, %lf ms, %lf ms:", data[0], data[1], data[2])
    for i in range(len(data)):
        data[i] = (data[i] - min_delay) / (max_delay - min_delay)
        if data[i] <= 0:
            data[i] = 0
    delay_pre = evaluate(lstm_model, np.array(data).reshape(1, len(data), 1))
    Logger.logger.debug("The predict downlink transmission delay based on OAI is: %lf ms", delay_pre)
    delay_pre = delay_pre // STEP + 1
    return delay_pre

if __name__ == '__main__':
    lag = 3
    seq = 1
    epoch = 10
    batch = 2
    neurons = 30
    scaler_supervised_oai = deal_data_supervised(lag, seq)
    n_test = int(scaler_supervised_oai.shape[0] * 0.01)
    trained_model, _, _, _, _, _ = train_lstm(data=scaler_supervised_oai, test_flag=n_test, n_input=lag, n_batch=batch, n_epoch=epoch, n_neurons=neurons)
    save_lstm_model(trained_model)
