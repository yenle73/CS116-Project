import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import os
import numpy as np

def load_data(data_directory):
    dataset = data_directory.split('/')
    df = pd.read_csv(f"{data_directory}/{dataset[1]}.csv")
    return df

def preprocess_data(data):
    # Tách features (X) và targets (y)
    X = data.drop(columns=['RoadSlope_100ms', 'Vehicle_Mass'])
    y = data[['RoadSlope_100ms', 'Vehicle_Mass']]

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def feature_engineering(X):
  X['Combined_VehV_Rng'] = 1 / X['VehV_v_100ms'] * X['RngMod_trqCrSmin_100ms']
  X['Combined_VehV_v_100ms_ActMod_trqInr_100ms'] = 1 / X['VehV_v_100ms'] * X['ActMod_trqInr_100ms']
  return X

