from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import pandas as pd
from exp.exp_basic import Exp_Basic
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Exp_SVR(object):
    
    def __init__(self, args):
        self.args = args
        self.model = self._build_model()
        data, labels = self._get_data("train")
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        self.X_train = data[0:border2s[0]]
        self.y_train = labels[0:border2s[0]]
        self.X_test = data[border2s[0]:border2s[1]]
        self.y_test = labels[border2s[0]:border2s[1]]
        self.X_val = data[border2s[1]:border2s[2]]
        self.y_val = labels[border2s[1]:border2s[2]]
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        # self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)

    def _build_model(self):
    # Here you could choose different kernels like 'linear', 'poly', 'rbf', 'sigmoid', etc.
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        return model

    def _select_criterion(self):
        criterion =  mean_squared_error
        return criterion

    def _get_data(self, flag):
        file_path = r'data\ETT\ETTh1.csv'
        data = pd.read_csv(file_path)
        data_features = data.drop(columns=['date'])
        n_steps = 40
        n_features = len(data_features.columns) - 1
        sequence_data = []
        sequence_labels = []
        for i in range(n_steps, len(data_features)):
            # Extract the sequence from prior steps
            end_ix = i
            start_ix = end_ix - n_steps
            sequence = data_features.iloc[start_ix:end_ix, :-1].values.flatten()  # Flatten the matrix to a vector
            sequence_data.append(sequence)
            sequence_labels.append(data_features.iloc[end_ix, -1])  # Get the corresponding label
        sequence_data = np.array(sequence_data)
        sequence_labels = np.array(sequence_labels)
        scaler = StandardScaler()
        sequence_data_scaled = scaler.fit_transform(sequence_data)
        print(f"Data shape: {sequence_data_scaled.shape}, Labels shape: {sequence_labels.shape}")
        return sequence_data_scaled, sequence_labels

    def train(self, setting, test_size=0.2, random_state=42):
        # Load and split the data
        data, labels = self._get_data("train")
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        print(f'Training completed. Mean Squared Error on test set: {mse}, Mean Absolute Error on test set: {mae}')

    def test(self, setting):
        X_val, y_val = self.X_val, self.y_val
        predictions = self.model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        mae = mean_absolute_error(y_val, predictions)
        print(f'Testing completed. Mean Squared Error on validation set: {mse}, Mean Absolute Error on validation set: {mae}')
