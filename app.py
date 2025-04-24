from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from functools import lru_cache
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
import requests

app = Flask(__name__)
CORS(app)

COIN_MAP = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "usdt": "USDT-USD",
    "ton": "TON11419-USD"  # Заменить на актуальный тикер, если потребуется
}

CURRENCY_SYMBOLS = {
    "usd": "$", "rub": "₽", "kzt": "₸", "eur": "€",
    "byn": "Br", "uah": "₴", "chf": "₣", "gbp": "£"
}


@lru_cache(maxsize=32)
def get_exchange_rate(target_currency):
    try:
        if target_currency == "usd":
            return 1.0

        url = f"https://api.exchangerate.host/convert?from=USD&to={target_currency.upper()}"
        response = requests.get(url)
        data = response.json()

        if "result" not in data:
            print(f"[ERROR] Invalid response from API: {data}")
            return None

        return data["result"]
    except Exception as e:
        print(f"[ERROR] Currency fetch failed: {e}")
        return None


@lru_cache(maxsize=16)
def get_crypto_data(coin_id, days=365):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        ticker = COIN_MAP.get(coin_id.lower())
        if not ticker:
            return None
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Close'].values.reshape(-1, 1) if not data.empty else None
    except Exception as e:
        print(f"[ERROR] Crypto data fetch failed: {e}")
        return None


def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


@app.route('/predict/<coin_id>/<int:days>', methods=['GET'])
def predict(coin_id, days):
    currency = request.args.get("currency", "usd").lower()
    if currency not in CURRENCY_SYMBOLS:
        return jsonify({"error": "Неверная валюта"}), 400

    prices = get_crypto_data(coin_id)
    if prices is None or len(prices) < 61:
        return jsonify({"error": f"Недостаточно данных для {coin_id}"}), 400

    try:
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
        model.fit(X, y, epochs=60, batch_size=64, verbose=0)

        predictions = []
        last_sequence = scaled_prices[-time_step:]
        for _ in range(days):
            last_sequence = last_sequence.reshape((1, time_step, 1))
            pred_scaled = model.predict(last_sequence, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(float(pred))
            last_sequence = np.append(last_sequence[:, 1:, :], [[pred_scaled[0]]], axis=1)

        exchange_rate = get_exchange_rate(currency)
        if exchange_rate is None:
            return jsonify({"error": "Не удалось получить курс валюты"}), 500

        converted_predictions = [round(p * exchange_rate, 2) for p in predictions]
        dates = [(datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(days)]

        return jsonify({
            "currency": currency.upper(),
            "symbol": CURRENCY_SYMBOLS[currency],
            "predictions": list(zip(dates, converted_predictions))
        })

    except Exception as e:
        return jsonify({"error": f"Ошибка предсказания: {str(e)}"}), 500


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


@lru_cache(maxsize=32)
def get_exchange_rate(target_currency):
    try:
        if target_currency == "usd":
            return 1.0

        url = f"https://api.exchangerate.host/convert?from=USD&to={target_currency.upper()}"
        response = requests.get(url)
        data = response.json()

        if "result" not in data:
            print(f"[ERROR] Invalid response from API: {data}")
            return None

        return data["result"]
    except Exception as e:
        print(f"[ERROR] Currency fetch failed: {e}")
        return None


