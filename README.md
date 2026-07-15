# Air Quality Index Prediction Using Time Series LSTM with Attention Mechanism and Autoencoder

**Student Name:** Ramachandra J  
**Roll Number:** 727824tuam036  

## Problem Statement
Rising air pollution levels in urban areas require accurate forecasting to enable timely public health advisories, but traditional statistical models struggle with complex temporal pollution patterns. This project develops an LSTM-based time series model with an attention mechanism and autoencoder to predict the Air Quality Index accurately.

## Architecture Diagram
![Architecture Diagram](docs/architecture.png)

## Dataset Source
The dataset used is the **Air Quality Dataset** from the UCI Machine Learning Repository. It contains responses from a gas multisensor device deployed in an Italian city, recording hourly instances.
**Link:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd DL_AQI_Prediction_727824tuam036
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preprocessing & EDA:**
   - Run the Jupyter Notebook `notebooks/EDA.ipynb` to visualize the dataset and understand the distributions.
   - The preprocessing logic is contained in `src/preprocess.py`.

4. **Train the Model:**
   ```bash
   python src/train.py
   ```
   This will download the data (if not already downloaded), preprocess it, build the Autoencoder-LSTM-Attention model, train it, and save the best model weights to `models/best_model.h5`.

5. **Evaluate the Model:**
   - Run `notebooks/Evaluation.ipynb` to see the plots of the training/validation loss and evaluate the model using time-series metrics.
   - Alternatively, run `python src/predict.py` for inference.

6. **Web Application (Streamlit):**
   - We have built an interactive web application for real-time predictions.
   - Run `streamlit run app.py` to start the web app.
   - You can randomly select 24-hour sequences from the test set and visually compare the predicted next hour AQI vs the true value.

## Results
| Metric | Value |
| --- | --- |
| MAE (Mean Absolute Error) | *To be updated* |
| RMSE (Root Mean Squared Error) | *To be updated* |
| R² Score | *To be updated* |

## Module Mapping
- **Module 1 (Sequence Models):** An **LSTM (Long Short-Term Memory)** network is used as the core sequence learning mechanism to capture long-term temporal dependencies in the air quality data.
- **Module 2 (Advanced Architectures):** An **Autoencoder** is employed for initial feature extraction and noise reduction from the raw sensor readings. An **Attention Mechanism** is integrated to allow the model to dynamically weigh the importance of different past time steps when forecasting the future AQI.
- **Module 3 (Real-World Impact):** The model is applied to a real-world environmental dataset, demonstrating the practical application of deep learning in predicting and managing urban air pollution for public health.

## References
1. *To be updated after Literature Survey*
2. *To be updated after Literature Survey*
3. *To be updated after Literature Survey*
4. *To be updated after Literature Survey*
5. *To be updated after Literature Survey*

## Demo
*Demo screenshots or GIF will be added here*
