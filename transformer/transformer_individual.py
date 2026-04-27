import matplotlib.pyplot as plt
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

from codecarbon import EmissionsTracker

import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from keras import Model, Input, layers

print(tf.config.list_physical_devices('GPU'))

# -----------------------
# Global reproducibility
# -----------------------
SEED = 12019844

# Python
os.environ["PYTHONHASHSEED"] = str(SEED)

# Python built-in RNG
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# TensorFlow
tf.random.set_seed(SEED)

# (optional but recommended for full determinism)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "TRANSFORMER"

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

MODELS_DIR = os.path.join("transformer", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


LOSS = 'mae'
BATCH_SIZE = 24
EPOCHS = 100
LEARNING_RATE = 0.001 # to be tuned
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5


def transformer_encoder(x, head_size, num_heads, ff_dim, dropout):
    # Attention block
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x_norm, x_norm)
    attn = layers.Dropout(dropout)(attn)
    x = x + attn  # residual

    # Feed-forward block
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x_norm)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Conv1D(filters=x.shape[-1], kernel_size=1)(ff)

    return x + ff  # residual


def model_builder(hp):
    # -----------------------
    # Hyperparameters (SLIM)
    # -----------------------
    head_size = hp.Choice("head_size", [8, 16])
    num_heads = hp.Choice("num_heads", [2, 4])
    ff_dim = hp.Choice("ff_dim", [16, 32])
    num_blocks = hp.Choice("num_transformer_blocks", [1, 2])
    dropout = hp.Choice("dropout", [0.1, 0.2])

    mlp_units = hp.Choice("mlp_units", [32, 64])
    mlp_dropout = hp.Choice("mlp_dropout", [0.1, 0.2])

    lr = hp.Choice("lr", [1e-3, 1e-4])

    # -----------------------
    # Input
    # -----------------------
    inputs = Input(shape=(WINDOW_LEN, len(input_cols)))

    # -----------------------
    # Embedding
    # -----------------------
    emb_dim = len(input_cols) * 2  # simple + stable choice
    x = layers.Dense(emb_dim)(inputs)

    # -----------------------
    # Transformer blocks
    # -----------------------
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # -----------------------
    # Head
    # -----------------------
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(mlp_units, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(len(FORECAST_DAYS))(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=LOSS
    )

    return model



early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

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
    
    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        max_epochs=EPOCHS,
        directory="tuning",
        project_name=f"{MODEL}_{spring_id}_{experiment_timestamp}",
        overwrite=True,
        seed=SEED
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    model = tuner.get_best_models(1)[0]

    emissions_train = tracker.stop()
    energy_kwh_train = tracker.final_emissions_data.energy_consumed

    model.save(os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.keras"))

    best_hp = tuner.get_best_hyperparameters(1)[0]

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
        plt.title(f"{MODEL} {spring_id} – {d}-Day Ahead Forecast")
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
            "energy inference [kWh]": energy_kwh_inference,
            "head_size": best_hp.get("head_size"),
            "num_heads": best_hp.get("num_heads"),
            "ff_dim": best_hp.get("ff_dim"),
            "num_transformer_blocks": best_hp.get("num_transformer_blocks"),
            "dropout": best_hp.get("dropout"),
            "mlp_units": best_hp.get("mlp_units"),
            "mlp_dropout": best_hp.get("mlp_dropout"),
            "learning_rate": best_hp.get("lr")
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILEPATH, index=False)
