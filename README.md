# Stacked LSTM for Time-Series Stock Forecasting

## Overview
This project explores the application of Long Short-Term Memory (LSTM) neural networks in predicting financial time-series data. The model is trained to analyze historical stock market trends and generate a continuous 30-day future price forecast. 

Rather than focusing solely on minimizing theoretical loss, this project critically examines the architectural behavior of deep sequence models when applied to highly volatile, real-world financial data.

## Data Acquisition & Preprocessing
* **Asset:** S&P 500 ETF Trust (SPY)
* **Timeframe:** January 2010 to Present (14+ years)
* **Source:** Tiingo API
* **Preprocessing:** The daily closing prices were extracted, reshaped, and normalized using `MinMaxScaler` to scale values between 0 and 1, facilitating optimal gradient descent during model training. The data was split into a 65% training set and a 35% testing set.

## Model Architecture
The network is built using TensorFlow/Keras and relies on a "Many-to-One" configuration, utilizing a sliding look-back window of 100 days to predict the subsequent day's price.

* **Input Layer:** `(100, 1)` representing the 100-day chronological sequence.
* **Hidden Layers:**
  * LSTM (50 Units, `return_sequences=True`)
  * LSTM (50 Units, `return_sequences=True`)
  * LSTM (50 Units, `return_sequences=False`)
* **Output Layer:** Dense (1 Unit)
* **Compilation:** Adam Optimizer, Mean Squared Error (MSE) Loss.

## Results and Evaluation
The model successfully converged and was evaluated against the unseen 35% test set. The predictions were inverse-transformed to calculate error metrics in real dollar values.

* **Mean Absolute Percentage Error (MAPE):** ~4.08%
* **Directional Variance:** 95.92
  
## Critical Analysis & Limitations
While the model achieves a low percentage error on historical data, the 30-day autoregressive future forecast effectively highlights a fundamental limitation of classical LSTM architectures in quantitative finance: **Volatility Smoothing**.

When executing the 30-day `while` loop forecast, the model is forced to feed its own predictions back into the input sequence. Because the network cannot predict random daily market shocks, the compounding uncertainty forces the output to regress toward a smoothed, linear trend (a moving average). 

This behavioral observation validates the industry shift away from standard RNN/LSTM structures and toward tokenized, transformer-based foundation models for long-horizon time-series forecasting.

## Usage
To run this notebook locally without requiring an API key:
1. Clone this repository.
2. Ensure `SPY.csv` is located in the same directory as the notebook.
3. Execute the Jupyter Notebook cells sequentially.
