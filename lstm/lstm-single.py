import os
import sys
import pickle
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.saving import load_model
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *


SEED = 12019844
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


TRAIN_PATH = '/Users/peter/pCloud Sync/TU/DataScience/Masterarbeit/Data/ma-datawrangling-v2/springs/395012/395012_train.csv'
VALID_PATH = '/Users/peter/pCloud Sync/TU/DataScience/Masterarbeit/Data/ma-datawrangling-v2/springs/395012/395012_valid.csv'
TEST_PATH = '/Users/peter/pCloud Sync/TU/DataScience/Masterarbeit/Data/ma-datawrangling-v2/springs/395012/395012_test.csv'

SCALER_Y_PATH = '/Users/peter/pCloud Sync/TU/DataScience/Masterarbeit/Data/ma-datawrangling-v2/springs/395012/395012_scale_y.pkl'

LOAD=False
MODEL_PATH = os.path.join("lstm", "lstm.keras")


# Model params
BATCH_SIZE = 24
DROPOUT = 0.1  # to be tuned
RECURRENT_DROPOUT_RATE = 0.3 # to be tuned
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EPOCHS = 100
LOSS = 'mae'
LSTM_UNITS = 96  # to be tuned
N_DENSE_LAYERS = 0  # to be tuned
N_STEPS = 96 * 4  # Window length (hours), to be tuned? *4 because of 15 min interval
LEARNING_RATE = 0.001

TARGET_COL = 'discharge'

FORECAST_DAYS = [1, 2, 3, 4]
FORECAST_HOURS = [d * 24 for d in FORECAST_DAYS]
FORECAST_15MS = [d * 24 * 4 for d in FORECAST_DAYS]
N_OUTPUTS = len(FORECAST_DAYS)

print("Reading training data...")
train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
print("Reading validation data...")
valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
print("Reading test data...")
test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])


def create_sequences(X, y, window_len, forecast_steps):
    Xs, ys = [], []
    max_horizon = max(forecast_steps)

    for i in range(len(X) - window_len - max_horizon + 1):
        Xs.append(X.iloc[i:i + window_len].values)
        targets = [y.iloc[i + window_len + h - 1] for h in forecast_steps]
        ys.append(targets)

    return np.array(Xs), np.array(ys)

input_cols = [c for c in train_df.columns if c not in ['timestamp']]
print("Creating sequences for training data...")
X_train, y_train = create_sequences(train_df[input_cols], train_df[TARGET_COL], N_STEPS, FORECAST_15MS)
print("Creating sequences for validation data...")
X_valid, y_valid = create_sequences(valid_df[input_cols], valid_df[TARGET_COL], N_STEPS, FORECAST_15MS)
print("Creating sequences for test data...")
X_test, y_test  = create_sequences(test_df[input_cols], test_df[TARGET_COL],  N_STEPS, FORECAST_15MS)


if LOAD:
    model = load_model(MODEL_PATH)
else:
    # Model Architecture
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(N_STEPS, len(input_cols)), dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT_RATE),
        Dense(N_OUTPUTS)
    ])
    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS)

    # Early stopping
    early_stopping = EarlyStopping(monitor=EARLY_STOPPING_MONITOR, patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    print("Training...")
    # Train
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
        verbose=1
    )

    print("Saving model...")
    model.save(MODEL_PATH)

# Predict
print("Predicting...")
y_pred = model.predict(X_test)

with open(SCALER_Y_PATH, 'rb') as f:
    y_scaler = pickle.load(f)

# Inverse transform to original scale and convert to m^3/s
y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

base_idx = N_STEPS + max(FORECAST_15MS) - 1
timestamps = test_df["timestamp"].iloc[base_idx:].values

# Evaluation
for i, d in enumerate(FORECAST_DAYS):
    print(f"\n=== {d}-Day Ahead ===")
    y_target_d = y_test_orig[:, i]
    y_pred_d = y_pred_orig[:, i]
    print(evaluate_forecast(y_target_d, y_pred_d))

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, y_test_orig[:, i], label="Observed", linewidth=1)
    plt.plot(timestamps, y_pred_orig[:, i], label="Predicted", linewidth=1)

    plt.title(f"Aubachquelle – {d}-Day Ahead Forecast")
    plt.xlabel("Time")
    plt.ylabel("Discharge [m³/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()


print("Done")
