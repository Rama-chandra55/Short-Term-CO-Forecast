import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from src.preprocess import preprocess_pipeline
from src.model import build_hybrid_model

def plot_history(history, save_path="assets/training_curves.png"):
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (MSE)')
    plt.legend()
    
    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def train():
    print("Starting preprocessing...")
    seq_length = 24
    X_train, X_test, y_train, y_test, scaler, target_idx = preprocess_pipeline(seq_length=seq_length)
    
    n_features = X_train.shape[2]
    
    print("Building model...")
    model = build_hybrid_model(seq_length, n_features)
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    checkpoint = ModelCheckpoint("models/best_model.h5", 
                                 monitor="val_loss", 
                                 save_best_only=True,
                                 verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    print("Training model...")
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint, early_stop])
    
    print("Training complete. Plotting curves...")
    plot_history(history)
    
    # Evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

if __name__ == "__main__":
    train()
