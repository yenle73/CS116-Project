import argparse
import os
from preprocess_data import load_data, preprocess_data, feature_engineering
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
import json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def train_regression_model(X_train, y_slope_train):
    model = KNeighborsRegressor(leaf_size=10, n_neighbors=1, p=1, weights='distance')
    model.fit(X_train, y_slope_train)
    return model

def train_classification_model(X_train, y_weight_train):
    model = BaggingClassifier(DecisionTreeClassifier(random_state=42))
    model.fit(X_train, y_weight_train)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train and predict using models')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], help='Specify the mode (train/predict)')
    parser.add_argument('--train_dir', type=str, default='public_data/train', help='Path to the training data')
    parser.add_argument('--dev_dir', type=str, default='public_data/test', help='Path to the development data')
    parser.add_argument('--test_dir', type=str, default='public_data/test', help='Path to the test data')
    parser.add_argument('--model_output_dir', type=str, default='saved_model', help='Directory to save/load the model')
    parser.add_argument('--output_file', type=str, default='predictions.json', help='Output file for predictions')
    args = parser.parse_args()

    if args.mode == 'train':
        os.makedirs(args.model_output_dir, exist_ok=True)
        # Load and preprocess training data
        train_data = load_data(args.train_dir)
        X_train, y_train = preprocess_data(train_data)
        X_train = feature_engineering(X_train)

        rb_scaler = RobustScaler().fit(X_train)
        X_train = rb_scaler.transform(X_train)

        y_train_slope = y_train['RoadSlope_100ms']
        y_train_mass = y_train['Vehicle_Mass']

        # Train and save models
        model_slope = KNeighborsRegressor(leaf_size=10, n_neighbors=1, p=1, weights='distance')
        model_slope.fit(X_train, y_train_slope)

        model_weight = BaggingClassifier(DecisionTreeClassifier(random_state=42))
        model_weight.fit(X_train, y_train_mass)

        model_weight_path = os.path.join(args.model_output_dir, 'trained_weight.joblib')
        model_slope_path = os.path.join(args.model_output_dir, 'trained_slope.joblib')
        rb_scaler_path = os.path.join(args.model_output_dir, 'rb_scaler.joblib')

        joblib.dump(rb_scaler, rb_scaler_path)
        joblib.dump(model_weight, model_weight_path)
        joblib.dump(model_slope, model_slope_path)

    elif args.mode == 'predict':
        # Load the trained model
        # Tải mô hình
        model_weight_path = os.path.join(args.model_output_dir, 'trained_weight.joblib')
        model_slope_path = os.path.join(args.model_output_dir, 'trained_slope.joblib')
        rb_scaler_path = os.path.join(args.model_output_dir, 'rb_scaler.joblib')

        model_weight = joblib.load(model_weight_path)
        model_slope = joblib.load(model_slope_path)
        rb_scaler = joblib.load(rb_scaler_path)

        # Load and preprocess test data
        X_test = load_data(args.test_dir)
        X_test = feature_engineering(X_test)
        X_test = rb_scaler.transform(X_test)
        # Make predictions
        slope_predictions = model_slope.predict(X_test)
        weight_predictions = model_weight.predict(X_test)

        pd.DataFrame({'RoadSlope_100ms':slope_predictions,'Vehicle_Mass':weight_predictions}).to_json(args.output_file, orient='records', lines=True)

    else:
        print("Invalid mode. Please use 'train' or 'predict'.")

if __name__ == "__main__":
    main()