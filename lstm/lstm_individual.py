import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

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

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "LSTM"

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

MODELS_DIR = os.path.join("lstm", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Model params
# Requirements for CUDNN: https://keras.io/api/layers/recurrent_layers/lstm/
BATCH_SIZE = 96 # 24 in original paper, here 24 * 4 for 15 min
DROPOUT = 0.1  # to be tuned
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EPOCHS = 100
LOSS = 'mae'
LSTM_UNITS = 96  # to be tuned
N_DENSE_LAYERS = 0  # to be tuned
LEARNING_RATE = 0.001 # to be tuned


with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

# for dev purposes
spring_ids = ["395012"]

for spring_id in spring_ids:
    print(f"Running {MODEL} for {spring_id}...")

    SPRING_DIR = os.path.join(SRINGS_BASE_DIR, spring_id)
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
                                       FORECAST_15MS)

    X_valid, y_valid, ts_valid  = create_sequences(valid_df[input_cols], 
                                       valid_df[TARGET_COL],
                                       valid_df["timestamp"], 
                                       WINDOW_LEN, 
                                       FORECAST_15MS)

    X_test, y_test, ts_test  = create_sequences(test_df[input_cols], 
                                       test_df[TARGET_COL],
                                       test_df["timestamp"], 
                                       WINDOW_LEN, 
                                       FORECAST_15MS)
      
    model = Sequential([
    LSTM(LSTM_UNITS, input_shape=(WINDOW_LEN, len(input_cols)), dropout=DROPOUT),
    Dense(len(FORECAST_DAYS))
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS)
    early_stopping = EarlyStopping(monitor=EARLY_STOPPING_MONITOR, patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    
    print("     Training...")
    
    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
        verbose=1
    )
    
    emissions_train = tracker.stop()
    energy_kwh_train = tracker.final_emissions_data.energy_consumed

    model.save(os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.keras"))


    print("     Inference...")

    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    y_pred = model.predict(X_test)

    emissions_inference = tracker.stop()
    energy_kwh_inference = tracker.final_emissions_data.energy_consumed

    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)

    # Inverse transform to original scale and convert to m^3/s
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

    for i, d in enumerate(FORECAST_DAYS):
        y_target_d = y_test_orig[:, i]
        y_pred_d = y_pred_orig[:, i]
        metrics = evaluate_forecast(y_target_d, y_pred_d)

        plots_base_dir = os.path.join(RESULTS_PLOTS_DIR, spring_id)
        os.makedirs(plots_base_dir, exist_ok=True)
        plot_filename = f"{spring_id}_{MODEL}_{d}d_{experiment_timestamp}.png"
        plot_path = os.path.join(plots_base_dir, plot_filename)

        timestamps_d = ts_test[:, i]

        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_d, y_test_orig[:, i], label="Observed", linewidth=1)
        plt.plot(timestamps_d, y_pred_orig[:, i], label="Predicted", linewidth=1)
        plt.title(f"{spring_id} – {d}-Day Ahead Forecast")
        plt.xlabel("Time")
        plt.ylabel("Discharge [m³/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        results.append({
            "spring_id": spring_id,
            "model": MODEL,
            "horizon": d,
            "nse": metrics["nse"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "smape": metrics["smape"],
            "emissions training [kg CO₂]": emissions_train,
            "energy training [kWh]": energy_kwh_train,
            "emissions inference [kg CO₂]": emissions_inference,
            "energy inference [kWh]": energy_kwh_inference
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILEPATH, index=False)
