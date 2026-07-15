import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from sklearn.preprocessing import MinMaxScaler
import pickle

def download_and_extract_data(data_dir="data"):
    """Downloads the UCI Air Quality dataset if not present."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    zip_path = os.path.join(data_dir, "AirQualityUCI.zip")
    csv_path = os.path.join(data_dir, "AirQualityUCI.csv")
    
    if not os.path.exists(csv_path):
        print("Downloading dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
    return csv_path

def load_and_clean_data(csv_path):
    """Loads CSV and handles missing values and formatting."""
    print("Loading data...")
    # The dataset uses ';' as separator and ',' as decimal point
    df = pd.read_csv(csv_path, sep=';', decimal=',', parse_dates=[['Date', 'Time']])
    
    # Drop rows/cols that are entirely NaN (artifact of the CSV format)
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Missing values are marked as -200
    df.replace(-200, np.nan, inplace=True)
    
    # Forward fill and backward fill for missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Convert Date_Time to datetime index
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y %H.%M.%S')
    df.set_index('Date_Time', inplace=True)
    
    return df

def create_sequences(data, target_col_idx, seq_length=24):
    """Creates sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)

def preprocess_pipeline(seq_length=24, target_col='AH'):
    """Complete preprocessing pipeline."""
    csv_path = download_and_extract_data()
    df = load_and_clean_data(csv_path)
    
    # Let's use 'AH' (Absolute Humidity) or 'PT08.S1(CO)' as target if we want continuous prediction.
    # We will predict CO(GT) representing AQI level essentially.
    target_col = 'CO(GT)'
    
    # Ensure numerical
    df = df.apply(pd.to_numeric, errors='coerce')
    df.ffill(inplace=True)
    
    target_col_idx = df.columns.get_loc(target_col)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    # Save scaler for later inverse transform
    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    X, y = create_sequences(scaled_data, target_col_idx, seq_length)
    
    # Train-test split (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, target_col_idx

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline()
    print("Preprocessing complete!")
