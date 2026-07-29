import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# 1. Download data


def download_data(ticker="MSFT", start="2015-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering 
# ---------------------------------------------------------------------------

def add_features(df):
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_30"] = out["Close"].rolling(30).mean()
    out["Volatility_10"] = out["Return"].rolling(10).std()

    # RSI (14-day)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    out = out.dropna()
    return out


FEATURE_COLS = ["Close", "Volume", "SMA_10", "SMA_30", "Volatility_10", "RSI_14"]
TARGET_COL = "Close"



# 3. Chronological split BEFORE scaling (prevents leakage)


def chronological_split(df, train_frac=0.8, val_frac=0.1):
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


# ---------------------------------------------------------------------------
# Scaling 
def fit_scalers(train_df):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_df[FEATURE_COLS])


    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_df[[TARGET_COL]])

    return feature_scaler, target_scaler


def apply_scalers(df, feature_scaler):
    scaled = df.copy()
    scaled[FEATURE_COLS] = feature_scaler.transform(df[FEATURE_COLS])
    return scaled


# ---------------------------------------------------------------------------
# 5. Windowing 
# ---------------------------------------------------------------------------

def make_windows(scaled_df, window_size=30):
    """
    Build (dates, X, y) where X[i] is the window_size x n_features block of
    features ending the day BEFORE the target day, and y[i] is the scaled
    Close on the target day.
    """
    feats = scaled_df[FEATURE_COLS].to_numpy()
    target = scaled_df[TARGET_COL].to_numpy()
    dates = scaled_df.index.to_numpy()

    X, y, out_dates = [], [], []
    for i in range(window_size, len(scaled_df)):
        X.append(feats[i - window_size:i])
        y.append(target[i])
        out_dates.append(dates[i])

    return (np.array(out_dates),
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# ---------------------------------------------------------------------------
# 6. Model
# ---------------------------------------------------------------------------

def build_model(window_size, n_features):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import layers, regularizers

    model = Sequential([
        layers.Input((window_size, n_features)),
        layers.LSTM(48, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(24, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001),
                  metrics=["mean_absolute_error"])
    return model


# ---------------------------------------------------------------------------
# 7. Metrics incl. naive baseline comparison
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    directional_acc = np.mean(true_dir == pred_dir) * 100

    print(f"[{label}] RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%  "
          f"DirAcc={directional_acc:.1f}%")
    return dict(rmse=rmse, mae=mae, mape=mape, directional_acc=directional_acc)


def naive_baseline(scaled_df, window_size, dates):
    """Predict tomorrow's Close = today's Close, on the original (unscaled) series."""
    close = scaled_df.loc[pd.to_datetime(dates), TARGET_COL]
    return close.shift(1).to_numpy()


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def main():
    WINDOW_SIZE = 30

    raw = download_data("MSFT", "2015-01-01", "2024-01-01")
    feat_df = add_features(raw)

    train_df, val_df, test_df = chronological_split(feat_df, 0.8, 0.1)
    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    feature_scaler, target_scaler = fit_scalers(train_df)

    train_scaled = apply_scalers(train_df, feature_scaler)
    val_scaled = apply_scalers(val_df, feature_scaler)
    test_scaled = apply_scalers(test_df, feature_scaler)

    # Windows need `window_size` days of history before the first prediction,
    # so pull that context from the tail of the previous split.
    def windows_with_context(prev_scaled, cur_scaled):
        context = prev_scaled.tail(WINDOW_SIZE)
        combined = pd.concat([context, cur_scaled])
        return make_windows(combined, WINDOW_SIZE)

    dates_train, X_train, y_train = make_windows(train_scaled, WINDOW_SIZE)
    dates_val, X_val, y_val = windows_with_context(train_scaled, val_scaled)
    dates_test, X_test, y_test = windows_with_context(val_scaled, test_scaled)

    print(f"Shapes -> X_train {X_train.shape}, X_val {X_val.shape}, "
          f"X_test {X_test.shape}")

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = build_model(WINDOW_SIZE, X_train.shape[2])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=32, callbacks=callbacks, verbose=1)

   
    def predict_prices(X):
        scaled_pred = model.predict(X).flatten()
        return target_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()

    train_pred = predict_prices(X_train)
    val_pred = predict_prices(X_val)
    test_pred = predict_prices(X_test)

    y_train_orig = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_orig = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    print("\n--- LSTM performance ---")
    evaluate(y_train_orig, train_pred, "Train")
    evaluate(y_val_orig, val_pred, "Val")
    evaluate(y_test_orig, test_pred, "Test")

    print("\n--- Naive baseline (predict tomorrow = today) ---")
    naive_test_pred = np.concatenate([[y_test_orig[0]], y_test_orig[:-1]])
    evaluate(y_test_orig[1:], naive_test_pred[1:], "Test (naive)")

    # Plots
    for name, dates, y_true, y_pred in [
        ("Train", dates_train, y_train_orig, train_pred),
        ("Validation", dates_val, y_val_orig, val_pred),
        ("Test", dates_test, y_test_orig, test_pred),
    ]:
        plt.figure(figsize=(12, 5))
        plt.plot(dates, y_true, label="Observed")
        plt.plot(dates, y_pred, label="Predicted")
        plt.title(f"{name}: Predictions vs. Observations (Original Scale)")
        plt.xlabel("Date")
        plt.ylabel("Close Price ($)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()