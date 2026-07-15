import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import AttentionLayer
from src.preprocess import preprocess_pipeline

st.set_page_config(page_title="AQI Prediction App", layout="wide")

st.title("Air Quality Index (AQI) Predictor")
st.markdown("### Predicting Carbon Monoxide (CO(GT)) levels using Autoencoder + LSTM + Attention Mechanism")

@st.cache_resource
def load_trained_model():
    model_path = "models/best_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    return None

@st.cache_resource
def get_data_and_scaler():
    # Only load a small batch of data for demo purposes
    X_train, X_test, y_train, y_test, scaler, target_idx = preprocess_pipeline(seq_length=24)
    return X_test, y_test, scaler, target_idx

model = load_trained_model()

if model is None:
    st.warning("⚠️ Model not found! Please run `python src/train.py` first to train and save the model.")
else:
    st.success("✅ Model loaded successfully!")
    
    with st.spinner("Loading test data..."):
        try:
            X_test, y_test, scaler, target_idx = get_data_and_scaler()
            
            st.markdown("### Make a Prediction")
            st.write("Click the button below to randomly select a 24-hour sequence from the test set and predict the next hour's AQI (CO).")
            
            if st.button("Predict Random Sequence"):
                # Pick a random sequence
                idx = np.random.randint(0, len(X_test))
                sample_seq = X_test[idx:idx+1]
                true_val = y_test[idx]
                
                # Predict
                pred_val = model.predict(sample_seq)[0][0]
                
                # Inverse transform logic (assuming we scaled all 15 features, we need a dummy array to inverse)
                dummy_true = np.zeros((1, scaler.n_features_in_))
                dummy_pred = np.zeros((1, scaler.n_features_in_))
                
                dummy_true[0, target_idx] = true_val
                dummy_pred[0, target_idx] = pred_val
                
                true_unscaled = scaler.inverse_transform(dummy_true)[0, target_idx]
                pred_unscaled = scaler.inverse_transform(dummy_pred)[0, target_idx]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("True CO(GT)", f"{true_unscaled:.2f}")
                col2.metric("Predicted CO(GT)", f"{pred_unscaled:.2f}")
                col3.metric("Absolute Error", f"{abs(true_unscaled - pred_unscaled):.2f}")
                
                # Plot the input sequence
                st.markdown("#### Input Sequence (24 Hours)")
                
                # Get unscaled sequence for the target column
                seq_scaled = sample_seq[0, :, target_idx]
                dummy_seq = np.zeros((24, scaler.n_features_in_))
                dummy_seq[:, target_idx] = seq_scaled
                seq_unscaled = scaler.inverse_transform(dummy_seq)[:, target_idx]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(1, 25), seq_unscaled, marker='o', label="Past 24h CO(GT)")
                ax.scatter(25, true_unscaled, color='green', s=100, label="True Next Hour", zorder=5)
                ax.scatter(25, pred_unscaled, color='red', marker='X', s=100, label="Predicted Next Hour", zorder=5)
                ax.set_title("24-Hour Context Window vs Prediction")
                ax.set_xlabel("Hour")
                ax.set_ylabel("CO(GT) Level")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error loading data or predicting: {str(e)}")
            st.info("Make sure you have an internet connection to download the UCI dataset if it hasn't been downloaded yet.")
