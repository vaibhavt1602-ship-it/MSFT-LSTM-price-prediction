# MSFT Stock Price Prediction with LSTM

A time-series forecasting pipeline that predicts next-day MSFT closing price using an LSTM neural network, built with a focus on avoiding common pitfalls in financial ML (data leakage, lack of baseline comparison, insufficient data usage).

## Overview

Stock price prediction with deep learning is easy to get subtly wrong — models can look like they're performing well while actually just learning that "tomorrow's price ≈ today's price" due to autocorrelation. This project builds an LSTM pipeline that is explicitly validated against a naive baseline to confirm it's extracting real signal, not just riding on that effect.

## Key Design Decisions

- **No data leakage**: the chronological train/val/test split happens *before* any scaler is fit, and scalers are fit only on training data.
- **Full historical range used**: windows the entire 2015–2024 dataset instead of a narrow slice, giving the model meaningfully more train/val/test samples.
- **Multivariate features**: instead of Close-price-only, the model uses Close, Volume, 10/30-day moving averages, rolling volatility, and RSI(14) — all computed causally (no lookahead).
- **Baseline comparison**: every run reports RMSE, MAE, MAPE, and directional accuracy for both the LSTM and a naive "predict tomorrow = today" baseline, so model quality can be judged honestly rather than by loss curves alone.
- **Regularization**: dropout and L2 weight regularization are used to control overfitting given the relatively small amount of daily stock data available.

## Pipeline

1. **Download** — pulls daily OHLCV data for MSFT via `yfinance`.
2. **Feature engineering** — adds Return, SMA_10, SMA_30, Volatility_10, RSI_14 (causal, rolling-window features only).
3. **Chronological split** — 80% train / 10% val / 10% test, split by date order (no shuffling, since order matters for time series).
4. **Scaling** — `MinMaxScaler` fit on training data only, applied to val/test to prevent leakage.
5. **Windowing** — builds sliding 30-day windows of features to predict the next day's Close.
6. **Model** — single-layer LSTM (48 units) + Dense layers with Dropout and L2 regularization.
7. **Training** — Adam optimizer, EarlyStopping, and ReduceLROnPlateau callbacks.
8. **Evaluation** — RMSE, MAE, MAPE, and directional accuracy, compared against a naive last-value baseline.

## Tech Stack

Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, yfinance, Matplotlib

## Usage

```bash
pip install tensorflow pandas numpy scikit-learn yfinance matplotlib
python msft_lstm_improved.py
```

## Results

 "LSTM achieved 44.8% directional accuracy on the test set vs. 0.0% for the naive baseline, with test RMSE of 
12.366
## Possible Extensions

- Predict returns instead of price levels
- Add market-wide features (S&P 500, VIX) as additional input channels
- Walk-forward (rolling-window) validation instead of a single fixed split
