# Time Series Forecasting - Bitcoin Price Analysis

## Overview

This repository contains a series of models designed to predict **Bitcoin prices** using historical time-series data. The models implement a range of machine learning and deep learning approaches, including **Dense Artificial Neural Networks (ANN)**, **Convolutional Neural Networks (Conv1D)**, **Long Short-Term Memory (LSTM)** networks, and more advanced methods like **N-Beats** and **Ensemble Models**. 

## Results

### Dataset Visualization:
<img src="https://github.com/leovidith/Bitcoin-Price-Prediction/blob/main/images/bitcoin%202.png" alt="Bitcoin Dataset Visualization" width="600"/>

### Performance Metrics:

Below are the performance metrics for each model:

| Model             | MAE        | MSE        | RMSE       | MAPE (%)   | MASE       |
|-------------------|------------|------------|------------|------------|------------|
| **Model 1: Dense ANN**    | 576.48     | 1,181,204  | 1,086.83   | 2.60       | 1.01       |
| **Model 2: Conv1D**      | 578.55     | 1,191,960  | 1,091.77   | 2.60       | 1.02       |
| **Model 3: LSTM**        | 581.04     | 1,218,118  | 1,103.68   | 2.62       | 1.02       |
| **Model 4: N-Beats**     | 579.60     | 1,158,628  | 1,076.40   | 2.65       | 1.02       |
| **Model 5: Ensemble**    | 577.34     | 1,166,218  | 1,079.92   | 2.60       | 1.01       |
| **Model 6: Turkey**      | 17,149.12  | 615,804,400| 24,815.41  | 121.62     | 26.54      |

### Visualizations:
<img src="https://github.com/leovidith/Bitcoin-Price-Prediction/blob/main/images/bitcoin.png" alt="Bitcoin Price Prediction" width="600"/>
<img src="https://github.com/leovidith/Bitcoin-Price-Prediction/blob/main/images/bitcoin1.png" alt="Bitcoin Price Prediction" width="600"/>

## Features

- **Model 1: Dense ANN (Artificial Neural Network)**: Fully connected feedforward neural network for predicting Bitcoin prices.
- **Model 2: Conv1D (1D Convolutional Neural Network)**: CNN-based model for detecting patterns in time-series data.
- **Model 3: LSTM (Long Short-Term Memory)**: A recurrent neural network designed for sequence prediction tasks, capturing long-term dependencies in data.
- **Model 4: N-Beats**: Deep learning-based time-series forecasting model that uses blocks of neural networks to capture trends and seasonal patterns.
- **Model 5: Ensemble**: Combines the predictions of multiple models (ANN, LSTM, Conv1D) to provide more accurate forecasts.
- **Model 6: Turkey Model**: Predicts Bitcoin prices considering extreme events, simulating catastrophic events in the market.

## Sprint Features

### Sprint 1: Data Preprocessing
- **Deliverable**: Cleaned and prepared Bitcoin time-series data, ready for model training.

### Sprint 2: Model Architecture and Training
- **Deliverable**: Trained models for Dense ANN, Conv1D, LSTM, N-Beats, Ensemble, and Turkey models.

### Sprint 3: Model Evaluation
- **Deliverable**: Evaluation of model performance based on MAE, MSE, RMSE, MAPE, and MASE metrics.

### Sprint 4: Results Visualization
- **Deliverable**: Visualizations showing the comparison of predicted vs. actual Bitcoin prices.

## Conclusion

The **Naive Bayes Model** provides the most robust approach by combining the strengths of individual models. However, the **Turkey Model** highlights how extreme events can disrupt predictions, with a very high error rate observed during its evaluation. This suggests the importance of accounting for such anomalies in time-series forecasting tasks. 
