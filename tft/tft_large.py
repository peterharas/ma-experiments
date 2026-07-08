import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
import random
import os

from datetime import datetime

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences_full_horizon

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tft.tft_custom_dataset import TFTCustomDataset
from tft.tft_train import train_tft

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import TorchNormalizer

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from codecarbon import EmissionsTracker

# ----------------------- REPRODUCABILITY -----------------------

SEED = 12019844
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Set seed for CUDA (if using GPUs)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
# Ensure deterministic behavior for PyTorch operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def cleanup_torch():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

# ----------------------- EXPERIMENT SETUP -----------------------

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "TFT_LARGE"
BATCH_SIZE = 24

FORECAST_LENGTH = 96

MODELS_DIR = os.path.join("tft", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LARGE_MODEL_DIR = os.path.join(MODELS_DIR, "large")
os.makedirs(LARGE_MODEL_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

# ----------------------- DATA PREPARATION -----------------------

with open(SPRING_LIST_FILE_TRAIN, 'r') as f:
    spring_ids_train = [line.strip() for line in f if line.strip()]

X_train_all, y_train_all = [], []
X_valid_all, y_valid_all = [], []

for spring_id in spring_ids_train:
    print(f"    Creating sequences for {spring_id}...")
    SPRING_DIR = os.path.join(SPRINGS_BASE_DIR, spring_id)
    TRAIN_PATH = os.path.join(SPRING_DIR, f"{spring_id}_train.csv")
    VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")

    if os.path.exists(TRAIN_PATH):
        train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
        input_cols = [c for c in train_df.columns if c not in ['timestamp']]

        X_train, y_train, _ = create_sequences_full_horizon(
            train_df[input_cols],
            train_df[TARGET_COL],
            train_df["timestamp"],
            WINDOW_LEN,
            FORECAST_LENGTH
        )
        X_train_all.append(X_train)
        y_train_all.append(y_train)
    
    if os.path.exists(VALID_PATH):
        valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
        input_cols = [c for c in valid_df.columns if c not in ['timestamp']]
        X_valid, y_valid, _ = create_sequences_full_horizon(
            valid_df[input_cols],
            valid_df[TARGET_COL],
            valid_df["timestamp"],
            WINDOW_LEN,
            FORECAST_LENGTH
        )
        X_valid_all.append(X_valid)
        y_valid_all.append(y_valid)

X_train_all = np.concatenate(X_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)
X_valid_all = np.concatenate(X_valid_all, axis=0)
y_valid_all = np.concatenate(y_valid_all, axis=0)

train_loader = DataLoader(
    TFTCustomDataset(X_train_all, y_train_all),
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    TFTCustomDataset(X_valid_all, y_valid_all),
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_input_features = len(input_cols)
# Define our variables
unknown_reals = [f"feat_{i}" for i in range(num_input_features)]
known_reals = ["dummy_known"]

# ----------------------- TUNING -----------------------

print("Tuning...")

config = {
    "hidden_size": tune.grid_search([64, 96]),
    "dropout": 0.1,
    "lr": tune.grid_search([0.001, 0.0001]),
    "lstm_layers": 2,
    "attention_head_size": 4,
    "epochs": 10,
    "patience": 2,
}

def trainable_tft(config, X_t=None, y_t=None, X_v=None, y_v=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TFTCustomDataset(X_t, y_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        TFTCustomDataset(X_v, y_v),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = TemporalFusionTransformer(
        hidden_size=config["hidden_size"],
        lstm_layers=config["lstm_layers"],
        hidden_continuous_size=config["hidden_size"],
        attention_head_size=config["attention_head_size"],
        output_size=1, # 1 for point prediction
        loss=MAE(),    
        output_transformer=TorchNormalizer(),
        max_encoder_length=WINDOW_LEN,
        x_reals=unknown_reals + known_reals,
        x_categoricals=[],
        time_varying_reals_encoder=unknown_reals + known_reals,
        time_varying_categoricals_encoder=[],
        time_varying_reals_decoder=known_reals,
        time_varying_categoricals_decoder=[],
        static_reals=[],
        static_categoricals=[],
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_tft(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        patience=config["patience"],
        use_ray=True,
        verbose=False
    )

scheduler = ASHAScheduler(
    max_t=config["epochs"],
    grace_period=config["patience"],
    reduction_factor=2,
)

trainable = tune.with_resources(
    tune.with_parameters(
        trainable_tft,
        X_t=X_train_all,
        y_t=y_train_all,
        X_v=X_valid_all,
        y_v=y_valid_all,
    ),
    resources={"cpu": 4, "gpu": 1}
)

tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        scheduler=scheduler,
        max_concurrent_trials=1,
    ),
    param_space=config,
)
tuning_results = tuner.fit()
best_result = tuning_results.get_best_result(metric="val_loss", mode="min")
best_config = best_result.config
print(best_config)

cleanup_torch()

# ----------------------- TRAINING -----------------------

print("Training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracker = EmissionsTracker(log_level="error")
tracker.start()


model = TemporalFusionTransformer(
    hidden_size=best_config["hidden_size"],
    lstm_layers=best_config["lstm_layers"],
    hidden_continuous_size=best_config["hidden_size"],
    attention_head_size=config["attention_head_size"],
    output_size=1, # 1 for point prediction
    loss=MAE(),
    output_transformer=TorchNormalizer(),
    max_encoder_length=WINDOW_LEN,
    x_reals=unknown_reals + known_reals,
    x_categoricals=[],
    time_varying_reals_encoder=unknown_reals + known_reals,
    time_varying_categoricals_encoder=[],
    time_varying_reals_decoder=known_reals,
    time_varying_categoricals_decoder=[],
    static_reals=[],
    static_categoricals=[],   
).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
MODEL_PATH = os.path.join(LARGE_MODEL_DIR, f"{MODEL}_{experiment_timestamp}.pt")

train_tft(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=config["epochs"],
    patience=config["patience"],
    model_save_path=MODEL_PATH,
)

emissions_train = tracker.stop()
energy_kwh_train = tracker.final_emissions_data.energy_consumed

# ----------------------- INFERENCE -----------------------

print("Inference...")

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

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
    X_test, y_test, ts_test  = create_sequences_full_horizon(test_df[input_cols], 
                                    test_df[TARGET_COL],
                                    test_df["timestamp"], 
                                    WINDOW_LEN, 
                                    FORECAST_LENGTH)

    test_loader = DataLoader(
        TFTCustomDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    # Extract indices: 23 (24h), 47 (48h), 71 (72h), 95 (96h)
    horizon_indices = [23, 47, 71, 95]

    preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            for k, v in batch_x.items():
                batch_x[k] = v.to(device)
                
            out = model(batch_x)
            full_preds = out.prediction.squeeze(-1).cpu() # Shape: (Batch, 96)
            
            target_preds = full_preds[:, horizon_indices] # Shape: (Batch, 4)
            
            preds.append(target_preds)

    y_pred = torch.cat(preds).numpy()

    emissions_inference = tracker.stop()
    energy_kwh_inference = tracker.final_emissions_data.energy_consumed

    # Inverse transform to original scale and convert to m^3/s
    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

    for i, (d, h_idx) in enumerate(zip(FORECAST_DAYS, horizon_indices)):
        # Slice the correct actual horizon (h_idx) and the correct prediction horizon (i)
        y_target_d = y_test_orig[:, h_idx]
        y_pred_d = y_pred_orig[:, i]
        metrics = evaluate_forecast(y_target_d, y_pred_d)

        plots_base_dir = os.path.join(RESULTS_PLOTS_DIR, spring_id)
        os.makedirs(plots_base_dir, exist_ok=True)
        plot_filename = f"{spring_id}_{MODEL}_{d}d_{experiment_timestamp}.png"
        plot_path = os.path.join(plots_base_dir, plot_filename)

        timestamps_d = ts_test[:, h_idx] 

        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_d, y_target_d, label="Observed", linewidth=1) 
        plt.plot(timestamps_d, y_pred_d, label="Predicted", linewidth=1)

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
            "hidden_size": best_config["hidden_size"],
            "learning_rate": best_config["lr"],
            "lstm_layers": best_config["lstm_layers"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILEPATH, index=False)

    cleanup_torch()
