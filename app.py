from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

cg = CoinGeckoAPI()

def get_bitcoin_data(days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = cg.get_coin_market_chart_range_by_id(
        id='bitcoin', vs_currency='usd',
        from_timestamp=int(start_date.timestamp()),
        to_timestamp=int(end_date.timestamp())
    )
    return [x[1] for x in data['prices']]

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Подготовка данных
prices = np.array(get_bitcoin_data()).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Модель LSTM
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
model.fit(X, y, epochs=60, batch_size=64, verbose=1)

@app.route('/predict/<int:days>', methods=['GET'])
def predict_get(days):
    predictions = []
    last_sequence = scaled_prices[-time_step:]

    for _ in range(days):
        last_sequence = last_sequence.reshape((1, time_step, 1))
        predicted_price_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
        predictions.append(predicted_price)
        last_sequence = np.append(last_sequence[:, 1:, :], [[predicted_price_scaled[0]]], axis=1)

    future_dates = [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]
    return jsonify({
        "predictions": [(str(date), float(price)) for date, price in zip(future_dates, predictions)]
    })

@app.route('/predict', methods=['POST'])
def predict_post():
    data = request.get_json()
    if not data or 'days' not in data:
        return jsonify({"error": "Invalid input"}), 400

    days = int(data['days'])
    return predict_get(days)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
