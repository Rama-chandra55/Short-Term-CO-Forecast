import tensorflow as tf
from src.model import AttentionLayer
from src.preprocess import preprocess_pipeline
import numpy as np

def predict():
    print("Loading data for prediction...")
    # Just to get the test set for a quick demo
    _, X_test, _, y_test, scaler, target_idx = preprocess_pipeline()
    
    print("Loading trained model...")
    try:
        model = tf.keras.models.load_model("models/best_model.h5", custom_objects={'AttentionLayer': AttentionLayer})
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by running src/train.py")
        return
        
    print("Running predictions...")
    preds = model.predict(X_test)
    
    # We would usually inverse_transform here, but since we scaled the whole matrix 
    # we would need to construct a dummy matrix to inverse transform.
    # For now, let's just print the scaled MAE
    mae = np.mean(np.abs(preds.flatten() - y_test))
    print(f"Prediction MAE on test set (scaled): {mae:.4f}")
    
    print("\nSample Predictions (Scaled):")
    for i in range(5):
        print(f"Actual: {y_test[i]:.4f}, Predicted: {preds[i][0]:.4f}")

if __name__ == "__main__":
    predict()
