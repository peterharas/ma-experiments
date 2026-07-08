import os
import random
import pickle
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences

from codecarbon import EmissionsTracker

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense

print(tf.config.list_physical_devices('GPU'))

# -----------------------
# Global reproducibility
# -----------------------
SEED = 12019844
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# ----------------------- EXPERIMENT SETUP -----------------------

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL = "LSTM_LARGE_ENERGY"

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

MODELS_DIR = os.path.join("lstm", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
LARGE_MODEL_DIR = os.path.join(MODELS_DIR, "large")
os.makedirs(LARGE_MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(LARGE_MODEL_DIR, f"{MODEL}_{experiment_timestamp}.keras")

# Path to the results file from your previous keras_tuner run for the large model
PREVIOUS_RESULTS_FILE = os.path.join(RESULTS_DIR, "LSTM_LARGE_results_20260504_063436.csv")
hp_df = pd.read_csv(PREVIOUS_RESULTS_FILE)

# Since it's a single large model, grab the hyperparameters from the first row
hp_row = hp_df.iloc[0]
lstm_units = int(hp_row["lstm_units"])
dropout = float(hp_row["dropout"])
lr = float(hp_row["learning_rate"])
n_dense_layers = int(hp_row["n_dense_layers"])

# Model params
BATCH_SIZE = 24
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EPOCHS = 100
LOSS = 'mae'

early_stopping = EarlyStopping(
    monitor=EARLY_STOPPING_MONITOR,
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

# ----------------------- DATA PREPARATION -----------------------

with open(SPRING_LIST_FILE_TRAIN, 'r') as f:
    spring_ids_train = [line.strip() for line in f if line.strip()]

X_train_all, y_train_all = [], []
X_valid_all, y_valid_all = [], []
input_cols = None

for spring_id in spring_ids_train:
    print(f"    Creating sequences for {spring_id}...")
    SPRING_DIR = os.path.join(SPRINGS_BASE_DIR, spring_id)
    TRAIN_PATH = os.path.join(SPRING_DIR, f"{spring_id}_train.csv")
    VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")
    
    if os.path.exists(TRAIN_PATH):
        train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
        if input_cols is None:
            input_cols = [c for c in train_df.columns if c not in ['timestamp']]
            
        X_train, y_train, _ = create_sequences(
            train_df[input_cols],
            train_df[TARGET_COL],
            train_df["timestamp"],
            WINDOW_LEN,
            FORECAST_HS
        )
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    if os.path.exists(VALID_PATH):
        valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
        X_valid, y_valid, _ = create_sequences(
            valid_df[input_cols],
            valid_df[TARGET_COL],
            valid_df["timestamp"],
            WINDOW_LEN,
            FORECAST_HS
        )
        X_valid_all.append(X_valid)
        y_valid_all.append(y_valid)

X_train_all = np.concatenate(X_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)
X_valid_all = np.concatenate(X_valid_all, axis=0)
y_valid_all = np.concatenate(y_valid_all, axis=0)

# ----------------------- TRAINING -----------------------

# Build Model
model = Sequential()
model.add(LSTM(lstm_units, input_shape=(WINDOW_LEN, len(input_cols)), dropout=dropout))
for _ in range(n_dense_layers):
    model.add(Dense(lstm_units, activation="relu"))
model.add(Dense(len(FORECAST_DAYS)))

model.compile(optimizer=Adam(learning_rate=lr), loss=LOSS)

print("     Training...")
tracker = EmissionsTracker(log_level="error")
tracker.start()

model.fit(
    X_train_all, y_train_all,
    validation_data=(X_valid_all, y_valid_all),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1
)

emissions_train = tracker.stop()
energy_kwh_train = tracker.final_emissions_data.energy_consumed

model.save(MODEL_PATH)

# ----------------------- INFERENCE -----------------------

print("     Inference...")

with open(SPRING_LIST_FILE_UNSEEN, 'r') as f:
    spring_ids_unseen = [line.strip() for line in f if line.strip()]

with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids_all = [line.strip() for line in f if line.strip()]

for spring_id in spring_ids_all:
    print(f"Evaluating {MODEL} for {spring_id}...")

    type_flag = "UNSEEN" if spring_id in spring_ids_unseen else "TRAIN"

    SPRING_DIR = os.path.join(SPRINGS_BASE_DIR, spring_id)
    TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
    SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")

    if not os.path.exists(TEST_PATH):
        print(f"    Skipping {spring_id} because of missing test data...")
        continue

    if not os.path.exists(SCALER_Y_PATH):
        VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")
        type_flag = "SEEN_MEANSCALING" if os.path.exists(VALID_PATH) else "UNSEEN_MEANSCALING"
        SCALER_Y_PATH = os.path.join(SPRINGS_BASE_DIR, "mean_scale_y.pkl")

    test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])
    X_test, y_test, ts_test = create_sequences(test_df[input_cols], test_df[TARGET_COL], test_df["timestamp"], WINDOW_LEN, FORECAST_HS)   

    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    y_pred = model.predict(X_test)

    emissions_inference = tracker.stop()
    energy_kwh_inference = tracker.final_emissions_data.energy_consumed

    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)

    # Inverse transform to original scale and convert to m^3/s
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred) * 0.001

    for i, d in enumerate(FORECAST_DAYS):
        y_target_d = y_test_orig[:, i]
        y_pred_d = y_pred_orig[:, i]
        metrics = evaluate_forecast(y_target_d, y_pred_d)

        results.append({
            "spring_id": spring_id,
            "spring_type": type_flag,
            "model": MODEL,
            "horizon": d,
            "nse": metrics["nse"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "smape": metrics["smape"],
            "emissions training [kg CO₂]": emissions_train,
            "energy training [kWh]": energy_kwh_train,
            "emissions inference [kg CO₂]": emissions_inference,
            "energy inference [kWh]": energy_kwh_inference,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "learning_rate": lr,
            "n_dense_layers": n_dense_layers
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILEPATH, index=False)