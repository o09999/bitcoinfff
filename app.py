# app.py
from flask import Flask, request, jsonify, render_template, current_app
from flask_cors import CORS
from functools import lru_cache
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
import requests
import joblib
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
CORS(app)

# --- Константы ---
COIN_MAP = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "usdt": "USDT-USD",
    "ton": "TON-USD"
}
CURRENCY_SYMBOLS = {
    "usd": "$", "rub": "₽", "kzt": "₸", "eur": "€",
    "byn": "Br", "uah": "₴", "chf": "₣", "gbp": "£"
}
TIME_STEP = 60
MODEL_SAVE_DIR = "models"
SCALER_SAVE_DIR = "scalers"
# --- ---

@lru_cache(maxsize=32)
def get_exchange_rate(target_currency):
    # ... (код функции get_exchange_rate остается БЕЗ ИЗМЕНЕНИЙ) ...
    try:
        target_currency = target_currency.upper()
        if target_currency == "USD":
            return 1.0
        api_url = f"https://api.frankfurter.app/latest?from=USD&to={target_currency}"
        current_app.logger.debug(f"Fetching exchange rate: {api_url}")
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "rates" in data and target_currency in data["rates"]:
            rate = data["rates"][target_currency]
            current_app.logger.debug(f"Rate for {target_currency}: {rate}")
            return float(rate)
        else:
            current_app.logger.error(f"Unexpected response structure from Frankfurter API for {target_currency}: {data}")
            return None
    except requests.exceptions.Timeout:
        current_app.logger.error(f"Currency fetch timed out for {target_currency} from {api_url}")
        return None
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Currency fetch failed for {target_currency}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        current_app.logger.error(f"Error processing currency data for {target_currency}: {e}")
        return None
    except Exception as e:
        current_app.logger.error(f"An unexpected error occurred in get_exchange_rate for {target_currency}: {e}")
        return None


@lru_cache(maxsize=16)
def load_model_and_scaler(coin_id):
     # ... (код функции load_model_and_scaler остается БЕЗ ИЗМЕНЕНИЙ) ...
    model_path = os.path.join(MODEL_SAVE_DIR, f"{coin_id}_model.keras")
    scaler_path = os.path.join(SCALER_SAVE_DIR, f"{coin_id}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        app.logger.error(f"Model or scaler file not found for {coin_id} at paths: {model_path}, {scaler_path}. Run train_models.py first.")
        return None, None
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        app.logger.info(f"Model and scaler loaded successfully for {coin_id}")
        return model, scaler
    except Exception as e:
        app.logger.error(f"Error loading model or scaler for {coin_id}: {e}", exc_info=True)
        return None, None

@app.route('/predict/<coin_id>/<int:days>', methods=['GET'])
def predict(coin_id, days):
    coin_id_lower = coin_id.lower()
    if coin_id_lower not in COIN_MAP:
         return jsonify({"error": f"Неизвестная криптовалюта: {coin_id}"}), 400

    if not (1 <= days <= 90):
        return jsonify({"error": "Количество дней должно быть от 1 до 90"}), 400

    # Сохраняем запрошенную валюту
    requested_currency = request.args.get("currency", "usd").lower()
    if requested_currency not in CURRENCY_SYMBOLS:
        return jsonify({"error": "Неверная валюта"}), 400

    ticker = COIN_MAP[coin_id_lower]
    app.logger.info(f"Prediction request for {coin_id_lower} ({ticker}), {days} days, requested currency {requested_currency.upper()}")

    # 1. Загрузка модели и скейлера
    model, scaler = load_model_and_scaler(coin_id_lower)
    if model is None or scaler is None:
        return jsonify({"error": f"Ошибка загрузки модели для {coin_id}. Пожалуйста, проверьте логи сервера."}), 500

    try:
        # 2. Получение последних данных
        # ... (код получения данных yfinance остается БЕЗ ИЗМЕНЕНИЙ) ...
        app.logger.info(f"Fetching last {TIME_STEP + 10} days data for {ticker} for prediction start...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=TIME_STEP + 15)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data.columns or len(data['Close']) < TIME_STEP:
             app.logger.error(f"Insufficient recent data for {ticker}. Needed >= {TIME_STEP}, got {len(data['Close'])} points.")
             return jsonify({"error": f"Недостаточно свежих данных ({ticker}) для начала предсказания."}), 400
        recent_prices = data['Close'].dropna().values.reshape(-1, 1)
        if len(recent_prices) < TIME_STEP:
             app.logger.error(f"Insufficient recent data for {ticker} after dropna. Needed >= {TIME_STEP}, got {len(recent_prices)} points.")
             return jsonify({"error": f"Недостаточно свежих данных ({ticker}) после очистки для начала предсказания."}), 400
        app.logger.info(f"Fetched {len(recent_prices)} recent valid points for {ticker}.")


        # 3. Подготовка входных данных
        # ... (код подготовки current_batch остается БЕЗ ИЗМЕНЕНИЙ) ...
        try:
            scaled_input_data = scaler.transform(recent_prices[-TIME_STEP:])
        except ValueError as ve:
             app.logger.error(f"ValueError during scaling for {ticker}. Data shape: {recent_prices[-TIME_STEP:].shape}. Error: {ve}")
             return jsonify({"error": f"Ошибка масштабирования данных для {ticker}. Возможно, проблема с форматом данных."}), 500
        current_batch = scaled_input_data.reshape((1, TIME_STEP, 1))


        # 4. Цикл предсказания
        # ... (код цикла предсказания остается БЕЗ ИЗМЕНЕНИЙ) ...
        predictions_scaled = []
        for i in range(days):
            pred_scaled_2d = model.predict(current_batch, verbose=0)
            pred_scaled = pred_scaled_2d[0, 0]
            predictions_scaled.append(pred_scaled)
            value_to_append = pred_scaled_2d.reshape(1, 1, 1)
            current_batch = np.append(current_batch[:, 1:, :], value_to_append, axis=1)


        # 5. Обратное масштабирование предсказаний
        # ... (код обратного масштабирования остается БЕЗ ИЗМЕНЕНИЙ) ...
        final_predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        final_predictions_usd = scaler.inverse_transform(final_predictions_scaled)
        app.logger.info(f"Generated {len(final_predictions_usd)} predictions in USD for {ticker}.")


        # 6. Конвертация в целевую валюту (с новой логикой)
        warning_message = None # Сообщение для пользователя, если курс недоступен
        response_currency_code = requested_currency # Валюта для ответа (по умолчанию запрошенная)
        response_symbol = CURRENCY_SYMBOLS[requested_currency] # Символ для ответа

        exchange_rate = get_exchange_rate(requested_currency)

        if exchange_rate is None:
            # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
            # Не удалось получить курс для запрошенной валюты
            app.logger.warning(f"Failed to get exchange rate for {requested_currency.upper()}. Defaulting prediction to USD.")
            # Формируем сообщение для пользователя
            warning_message = f"Не удалось получить прогноз в выбранной валюте ({requested_currency.upper()}), так как она временно не поддерживается или недоступна. Прогноз показан в USD."
            # Устанавливаем курс = 1.0 и валюту ответа = USD
            exchange_rate = 1.0
            response_currency_code = "usd"
            response_symbol = CURRENCY_SYMBOLS["usd"]
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        # else: # Если курс получен, exchange_rate, response_currency_code и response_symbol уже установлены правильно

        # Рассчитываем итоговые предсказания (либо в запрошенной валюте, либо в USD)
        converted_predictions = [round(float(p[0] * exchange_rate), 2) for p in final_predictions_usd]
        dates = [(datetime.now().date() + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(days)]

        app.logger.info(f"Successfully generated prediction for {coin_id}, {days} days. Displaying in {response_currency_code.upper()}.")

        # Возвращаем успешный ответ, добавляя поле warning (будет None, если все ок)
        return jsonify({
            "currency": response_currency_code.upper(), # Либо запрошенная, либо USD
            "symbol": response_symbol,              # Либо запрошенный, либо $
            "predictions": list(zip(dates, converted_predictions)),
            "warning": warning_message              # Сообщение или None
        }) # Код ответа по умолчанию 200 ОК

    except Exception as e:
        # Логируем любую другую ошибку во время предсказания
        app.logger.error(f"Prediction error for {coin_id}/{days}?currency={requested_currency}: {e}", exc_info=True)
        return jsonify({"error": f"Внутренняя ошибка сервера при предсказании: {str(e)}"}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)