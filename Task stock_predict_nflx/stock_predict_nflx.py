import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# === Load and preprocess dataset ===
df = pd.read_csv("nflx_2014_2023.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("Available columns:", df.columns)

# Check if required columns exist
required_cols = ['date', 'close']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV missing required column: {col}")

# Convert date column
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === Select multiple features for training ===
feature_cols = ['open', 'high', 'low', 'close', 'volume',
                'rsi_7', 'rsi_14', 'cci_7', 'cci_14',
                'sma_50', 'ema_50', 'sma_100', 'ema_100',
                'macd', 'bollinger', 'truerange', 'atr_7', 'atr_14']

features = df[feature_cols].values
target = df[['next_day_close']].values  # predict next day's close

# === Scale features ===
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_x = scaler_x.fit_transform(features)
scaled_y = scaler_y.fit_transform(target)

# === Create sequences ===
sequence_length = 60
x, y = [], []
for i in range(sequence_length, len(scaled_x)):
    x.append(scaled_x[i-sequence_length:i])
    y.append(scaled_y[i, 0])

x, y = np.array(x), np.array(y)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# === Build LSTM model ===
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("Training model... This may take a few minutes.")
model.fit(x_train, y_train, batch_size=32, epochs=20)

# === Predictions ===
predictions = model.predict(x_test)
predictions_rescaled = scaler_y.inverse_transform(predictions.reshape(-1, 1))
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# === Evaluation ===
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
r2 = r2_score(y_test_rescaled, predictions_rescaled)

print("\nModel Performance with Multiple Features:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# === Plot results ===
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, color="blue", label="Actual Prices")
plt.plot(predictions_rescaled, color="red", label="Predicted Prices")
plt.title("Netflix Stock Price Prediction (Multi-feature LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
