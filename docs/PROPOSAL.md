# Project Proposal

**Project Title:** Air Quality Index Prediction Using Time Series LSTM with Attention Mechanism and Autoencoder
**Student Name:** Ramachandra J
**Roll Number:** 727824tuam036
**Section / Year:** AIML - A/ 3rd year

## 1. Objective
Rising air pollution levels in urban areas require accurate forecasting to enable timely public health advisories. Traditional statistical models struggle with complex temporal pollution patterns. The primary objective of this project is to develop an advanced deep learning time series model using an Long Short-Term Memory (LSTM) network equipped with an Attention Mechanism and an Autoencoder to predict the Air Quality Index (AQI) accurately.

## 2. Dataset Source
The dataset will be sourced from the **UCI Machine Learning Repository** (specifically the Air Quality dataset, ID: 360). It contains over 9,000 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. 
Source: [https://archive.ics.uci.edu/dataset/360/air+quality](https://archive.ics.uci.edu/dataset/360/air+quality)

## 3. Methodology & Architecture Overview
The project will follow these methodological steps:
1.  **Data Preprocessing:** Handling missing values (-200 in the dataset), normalizing features using Min-Max scaling, and converting the data into sequential windows suitable for time-series forecasting.
2.  **Autoencoder (Module 2 concept):** An Autoencoder will be utilized for feature extraction and noise reduction. It compresses the input multivariate time series into a dense latent representation and reconstructs it, helping the model learn robust underlying patterns.
3.  **LSTM Network (Module 1 concept):** The latent representation or the raw sequences will be passed into an LSTM network, which is highly capable of learning long-term dependencies in time-series data.
4.  **Attention Mechanism (Module 2 concept):** An Attention layer will be added on top of the LSTM to allow the model to focus on the most relevant historical time steps when predicting the future AQI.
5.  **Output Layer:** A dense layer for the final regression output (AQI value or pollutant concentration).

## 4. Expected Outcome
The expected outcome is a trained deep learning model capable of forecasting AQI or pollutant levels with high accuracy. The model's performance will be evaluated using standard regression/time-series metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score. The attention weights will also provide interpretability regarding which past observations most heavily influence the predictions.
