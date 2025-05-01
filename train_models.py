# train_models.py
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
import joblib  # Для сохранения/загрузки скейлера
import os
import tensorflow as tf
import logging # Для логирования процесса

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Константы (можно вынести в отдельный config.py) ---
COIN_MAP = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "usdt": "USDT-USD",  # USDT-USD может иметь мало смысла для предсказания цены, т.к. он привязан к USD, но оставим для примера
    "ton": "TON-USD"      # Проверьте актуальный тикер для Toncoin на Yahoo Finance!
}
DAYS_HISTORY = 730 # Сколько дней истории использовать для обучения (например, 2 года)
TIME_STEP = 60     # Размер окна для LSTM (должно быть меньше DAYS_HISTORY)
EPOCHS = 50        # Количество эпох обучения (можно настроить)
BATCH_SIZE = 32    # Размер батча

MODEL_SAVE_DIR = "models"     # Папка для сохранения моделей Keras
SCALER_SAVE_DIR = "scalers"   # Папка для сохранения скейлеров
# --- ---

def create_dataset(data, time_step=60):
    """Создает датасет для LSTM из временного ряда."""
    X, y = [], []
    if len(data) <= time_step:
        logging.warning(f"Not enough data to create sequences with time_step={time_step}. Data length: {len(data)}")
        return np.array(X), np.array(y) # Возвращаем пустые массивы

    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train_and_save_model(coin_id, ticker):
    """Обучает и сохраняет модель LSTM и скейлер для одной криптовалюты."""
    logging.info(f"--- Training model for {coin_id} ({ticker}) ---")
    try:
        # 1. Загрузка данных
        logging.info(f"Fetching data for {ticker}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_HISTORY)
        # Используем auto_adjust=True для автоматической корректировки цен (сплиты, дивиденды)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if data.empty or 'Close' not in data.columns:
            logging.error(f"No data found or 'Close' column missing for {ticker}. Skipping.")
            return

        # Удаляем возможные NaN значения и проверяем длину
        prices = data['Close'].dropna().values.reshape(-1, 1)
        if len(prices) < TIME_STEP + 2: # Нужно хотя бы TIME_STEP + 2 точки для создания одного примера (TIME_STEP вход, 1 выход, 1 для цикла)
             logging.error(f"Not enough valid historical data for {ticker} after dropna ({len(prices)} points found, need >= {TIME_STEP + 2}). Skipping.")
             return
        logging.info(f"Data fetched and cleaned: {len(prices)} points for {ticker}.")

        # 2. Предобработка и создание датасета
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Обучаем скейлер и трансформируем данные
        scaled_prices = scaler.fit_transform(prices)

        X, y = create_dataset(scaled_prices, TIME_STEP)

        # Проверяем, создался ли датасет
        if X.shape[0] == 0 or y.shape[0] == 0:
             logging.error(f"Could not create training dataset for {ticker} (X shape: {X.shape}, y shape: {y.shape}). Insufficient data after sequencing? Skipping.")
             return

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        logging.info(f"Dataset created for {ticker}. X shape: {X.shape}, y shape: {y.shape}")

        # 3. Создание и обучение модели
        logging.info(f"Building LSTM model for {ticker}...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
            LSTM(50, return_sequences=False), # Последний LSTM слой не возвращает последовательность
            Dense(25),
            Dense(1) # Один выходной нейрон для предсказания цены
        ])
        # Компиляция модели
        model.compile(optimizer='adam', loss='mean_squared_error')

        logging.info(f"Training model for {ticker} for up to {EPOCHS} epochs...")
        # Используем EarlyStopping для предотвращения переобучения и сохранения лучшей модели
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',       # Следим за функцией потерь на обучающем наборе
            patience=10,          # Сколько эпох ждать улучшения перед остановкой (можно настроить)
            restore_best_weights=True # Восстановить веса модели из эпохи с лучшим значением monitor
        )
        # Обучаем модель
        history = model.fit(
            X,
            y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1, # Показываем прогресс обучения (можно поставить 0 для тишины или 2 для краткого лога)
            callbacks=[early_stopping], # Передаем early stopping
            shuffle=False # Для временных рядов часто лучше не перемешивать
        )
        # Логируем результат обучения
        final_loss = history.history['loss'][-1]
        best_epoch = np.argmin(history.history['loss']) + 1 # Эпоха с лучшей потерей
        logging.info(f"Training finished for {ticker}. Best epoch: {best_epoch}, Final loss: {final_loss:.6f}")

        # 4. Сохранение модели и скейлера
        # Создаем директории, если их нет (безопасно для повторного запуска)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(SCALER_SAVE_DIR, exist_ok=True)

        # Пути для сохранения
        # Используем формат .keras (рекомендуемый TensorFlow)
        model_path = os.path.join(MODEL_SAVE_DIR, f"{coin_id}_model.keras")
        scaler_path = os.path.join(SCALER_SAVE_DIR, f"{coin_id}_scaler.joblib")

        logging.info(f"Saving model to {model_path}")
        model.save(model_path) # Сохранение модели Keras

        logging.info(f"Saving scaler to {scaler_path}")
        joblib.dump(scaler, scaler_path) # Сохранение скейлера

        logging.info(f"--- Model and scaler for {coin_id} saved successfully ---")

    except Exception as e:
        # Логируем любую ошибку, возникшую во время обработки этой монеты
        logging.error(f"Failed to train model for {coin_id} ({ticker}): {e}", exc_info=True) # exc_info=True добавляет traceback

# --- Основной блок запуска скрипта ---
if __name__ == "__main__":
    logging.info("===== Starting model training process =====")
    # Перебираем все монеты из словаря COIN_MAP
    for coin, tick in COIN_MAP.items():
        train_and_save_model(coin, tick) # Вызываем функцию обучения для каждой монеты
    logging.info("===== Model training process finished =====")