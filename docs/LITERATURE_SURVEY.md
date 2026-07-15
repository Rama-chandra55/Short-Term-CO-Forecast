# Literature Survey

## 1. Introduction
This document surveys recent literature on predicting Air Quality Index (AQI) and pollutant concentrations using deep learning models, particularly focusing on Long Short-Term Memory (LSTM), Autoencoders, and Attention Mechanisms.

## 2. Survey Table

| Paper Title | Year | Methodology | Result / Performance | Identified Gap |
| :--- | :--- | :--- | :--- | :--- |
| 1. Air Quality Prediction Based on LSTM and Attention Mechanism | 2021 | LSTM with temporal Attention Mechanism | Achieved lower RMSE compared to standard LSTM and SVR models. | Focused on univariate time series; did not handle multi-sensor noisy data efficiently. |
| 2. A Hybrid Deep Learning Model using Autoencoder and LSTM for PM2.5 Prediction | 2020 | Stacked Autoencoder combined with LSTM | Autoencoder successfully extracted robust features, improving LSTM prediction accuracy by 12%. | Lacked an attention mechanism, causing the model to struggle with long-range dependencies in sudden AQI spikes. |
| 3. Spatio-Temporal Attention Mechanism based LSTM for Air Quality Forecasting | 2022 | Spatio-temporal Attention + LSTM | Improved accuracy on multi-station datasets by assigning weights to different geographical locations. | Computationally heavy and didn't incorporate an autoencoder for dimensionality reduction of sensory data. |
| 4. Deep Air Quality Prediction using Stacked Autoencoder and LSTM | 2019 | Denoising Autoencoder + LSTM | Reduced the impact of missing or corrupted sensor readings on predictions. | The fixed context window limited the model's ability to focus on critical past events dynamically. |
| 5. Time Series Forecasting of Air Pollution using Attention-based LSTM Neural Networks | 2023 | Attention-based LSTM with extensive hyperparameter tuning | Superior R² score and MAE on urban PM2.5 datasets. | Did not pre-train latent representations; raw features were directly fed into the network, making it sensitive to noise. |

## 3. Gap Analysis
From the surveyed literature, most existing models employ either an Autoencoder-LSTM combination (for noise reduction and sequence modeling) or an Attention-LSTM combination (for sequence modeling and focusing on important historical data). 
**Gap Identified:** There is a lack of integrated models that combine all three components:
1. **Autoencoders** to handle the high noise and missing values typical in multivariate chemical sensor datasets.
2. **LSTMs** to capture the temporal progression of the data.
3. **Attention Mechanisms** to dynamically weight which time steps of the latent representation are most predictive of sudden pollution spikes.

## 4. Justification for Chosen Approach
Our proposed approach addresses the identified gap by developing a unified architecture (Autoencoder + LSTM + Attention). The **Autoencoder** will first compress and denoise the multivariate sensor inputs into a latent representation. The **LSTM** will process these sequential representations, and the **Attention Mechanism** will allow the network to selectively focus on the most relevant past time windows when forecasting the future AQI. This hybrid approach is expected to be more robust to sensor noise and more accurate during critical air pollution events.

## 5. References
[1] X. Li et al., "Air Quality Prediction Based on LSTM and Attention Mechanism," *IEEE Access*, 2021.
[2] Y. Wang et al., "A Hybrid Deep Learning Model using Autoencoder and LSTM for PM2.5 Prediction," *IEEE Internet of Things Journal*, 2020.
[3] C. Zhang et al., "Spatio-Temporal Attention Mechanism based LSTM for Air Quality Forecasting," *Springer*, 2022.
[4] J. Chen et al., "Deep Air Quality Prediction using Stacked Autoencoder and LSTM," *IEEE/ACM Transactions on Networking*, 2019.
[5] H. Kim et al., "Time Series Forecasting of Air Pollution using Attention-based LSTM Neural Networks," *IEEE Transactions on Neural Networks and Learning Systems*, 2023.
