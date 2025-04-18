from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Сопоставление названий криптовалют с тикерами yfinance
COIN_MAP = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "usdt": "USDT-USD",
    "ton": "TON11419-USD"  # или замени на корректный тикер TON, если другой
}

def get_crypto_data(coin_id, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    ticker = COIN_MAP.get(coin_id.lower())
    if not ticker:
        return None
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1) if not data.empty else None

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Предсказание
@app.route('/predict/<coin_id>/<int:days>', methods=['GET'])
def predict(coin_id, days):
    prices = get_crypto_data(coin_id)
    if prices is None or len(prices) < 61:
        return jsonify({"error": f"Not enough data or invalid coin: {coin_id}"}), 400

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    time_step = 60
    X, y = create_dataset(scaled_prices, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=64, verbose=0)

    predictions = []
    last_sequence = scaled_prices[-time_step:]
    for _ in range(days):
        last_sequence = last_sequence.reshape((1, time_step, 1))
        predicted_price_scaled = model.predict(last_sequence, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
        predictions.append(predicted_price)
        last_sequence = np.append(last_sequence[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)

    future_dates = [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]
    return jsonify({
        "predictions": [(date, float(price)) for date, price in zip(future_dates, predictions)]
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
