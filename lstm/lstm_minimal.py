import pandas as pd
import pickle
import sys
import random

from datetime import datetime

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense

print(tf.config.list_physical_devices('GPU'))

SEED = 12019844
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "LSTM"

BATCH_SIZE = 24
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EPOCHS = 100
LOSS = 'mae'
LSTM_UNITS = 96
DROPOUT = 0.1
LR = 0.001

MODELS_DIR = os.path.join("lstm", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

# for dev purposes
# spring_ids = ["395012"]

for spring_id in spring_ids:
    print(f"Running {MODEL} for {spring_id}...")

    SPRING_DIR = os.path.join(SPRINGS_BASE_DIR, spring_id)
    TRAIN_PATH = os.path.join(SPRING_DIR, f"{spring_id}_train.csv")
    VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")
    TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
    SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VALID_PATH) or not os.path.exists(TEST_PATH) or not os.path.exists(SCALER_Y_PATH):
        print(f"    Skipping {spring_id} because of missing data")
        continue

    SPRING_MODEL_DIR = os.path.join(MODELS_DIR, spring_id)
    os.makedirs(SPRING_MODEL_DIR, exist_ok=True)

    print("     Reading data...")
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])

    print("     Creating sequences...")
    input_cols = [c for c in test_df.columns if c not in ['timestamp']]
    X_train, y_train, ts_train  = create_sequences(train_df[input_cols], 
                                       train_df[TARGET_COL],
                                       train_df["timestamp"], 
                                       WINDOW_LEN, 
                                       FORECAST_HS)

    X_valid, y_valid, ts_valid  = create_sequences(valid_df[input_cols], 
                                       valid_df[TARGET_COL],
                                       valid_df["timestamp"], 
                                       WINDOW_LEN, 
                                       FORECAST_HS)

    X_test, y_test, ts_test  = create_sequences(test_df[input_cols], 
                                       test_df[TARGET_COL],
                                       test_df["timestamp"], 
                                       WINDOW_LEN, 
                                       FORECAST_HS)
      

    
    print("     Training...")
    
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, input_shape=(WINDOW_LEN, len(input_cols)), dropout=DROPOUT))
    model.add(Dense(LSTM_UNITS, activation="relu"))
    model.add(Dense(len(FORECAST_DAYS)))
    model.compile(optimizer=Adam(learning_rate=LR),loss=LOSS)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
        verbose=1
    )

    model.save(os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.keras"))


    print("     Inference...")

    y_pred = model.predict(X_test)

    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)

    # Inverse transform to original scale and convert to m^3/s
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

    for i, d in enumerate(FORECAST_DAYS):
        y_target_d = y_test_orig[:, i]
        y_pred_d = y_pred_orig[:, i]
        metrics = evaluate_forecast(y_target_d, y_pred_d)

        print(metrics)
