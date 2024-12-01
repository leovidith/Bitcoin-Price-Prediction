# Bitcoin Price Predictor README

## Overview
This repository contains a series of models designed to predict Bitcoin prices using historical time-series data. Various machine learning and deep learning approaches are implemented, including basic models like Dense Artificial Neural Networks (ANN), Convolutional Neural Networks (Conv1D), Long Short-Term Memory networks (LSTM), and more advanced methods like N-Beats and Ensemble Models.

## Models Overview

### Model 1: Dense ANN (Artificial Neural Network)
- **Type**: Fully connected feedforward neural network.
- **Features**: Uses dense layers with ReLU activation to predict Bitcoin prices.
- **Training**: Trained on windowed time-series data (7 days window).
  
### Model 2: Conv1D Model (1D Convolutional Neural Network)
- **Type**: 1D Convolutional Neural Network (CNN).
- **Features**: Utilizes convolutional layers to detect patterns in time series data.
- **Training**: Similar to the Dense ANN but leverages the spatial features in sequences.

### Model 3: LSTM (Long Short-Term Memory)
- **Type**: Recurrent neural network (RNN) with LSTM units.
- **Features**: Designed to capture long-term dependencies in time-series data, ideal for sequence prediction tasks.
- **Training**: Trained on historical Bitcoin price data with 7-day windows.

### Model 4: N-BEATS Algorithm
- **Type**: Deep learning-based time-series forecasting model.
- **Features**: A neural network-based model designed for forecasting that doesn’t rely on domain knowledge. It uses blocks of neural networks to capture both trend and seasonal patterns.
- **Training**: Applies advanced stacking methods for accurate forecasting.

### Model 5: Ensemble Model
- **Type**: Combination of multiple models.
- **Features**: Uses the output of multiple models (e.g., Dense ANN, LSTM, Conv1D) to predict Bitcoin prices by averaging their predictions.
- **Training**: Multiple models are trained iteratively, and the final predictions are an ensemble of these models.

### Model 6: Turkey Model
- **Type**: A model designed to predict Bitcoin prices while considering an extreme event.
- **Features**: Applies a perturbation to the last data point to simulate catastrophic events and predict how Bitcoin's price reacts.
- **Training**: Trained on time-series data with adjustments to account for outlier or extreme events.

## Performance Metrics

Below are the performance metrics (evaluated on test data) for each model:

| Model             | MAE        | MSE        | RMSE       | MAPE (%)   | MASE       |
|-------------------|------------|------------|------------|------------|------------|
| **Model 1: Dense ANN**    | 576.48     | 1,181,204  | 1,086.83   | 2.60       | 1.01       |
| **Model 2: Conv1D**      | 578.55     | 1,191,960  | 1,091.77   | 2.60       | 1.02       |
| **Model 3: LSTM**        | 581.04     | 1,218,118  | 1,103.68   | 2.62       | 1.02       |
| **Model 4: N-Beats**     | 579.60     | 1,158,628  | 1,076.40   | 2.65       | 1.02       |
| **Model 5: Ensemble**    | 577.34     | 1,166,218  | 1,079.92   | 2.60       | 1.01       |
| **Model 6: Turkey**      | 17,149.12  | 615,804,400| 24,815.41  | 121.62     | 26.54      |

---

## How to Use

1. **Installation**
   - Clone this repository to your local machine.
   - Install the required dependencies using the following command:
     ```bash
     pip install -r requirements.txt
     ```

2. **Running the Models**
   - Execute the script to train and evaluate the models:
     ```bash
     python bitcoin_price_predictor.py
     ```

3. **Model Results**
   - After running the script, you will see the evaluation results printed on the console.
   - Visualizations of the predictions and actual prices will also be plotted.

---

## Conclusion

Each model has been evaluated on its ability to predict Bitcoin's price based on historical data. The ensemble model provides a robust approach by combining the strengths of individual models. However, the Turkey model demonstrates how extreme events can significantly disrupt predictions, as evidenced by its very high error metrics.
