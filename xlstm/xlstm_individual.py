import gc
import pickle
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences

from codecarbon import EmissionsTracker

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import xLSTMForecaster
from train import train_model

from ray import tune
from ray.tune.schedulers import ASHAScheduler

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

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Global model params
MODEL = "xLSTM"
BATCH_SIZE = 24
EARLY_STOPPING_PATIENCE = 3
EPOCHS = 100

MODELS_DIR = os.path.join("xlstm", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

def cleanup_torch():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

# for dev purposes
spring_ids = ["395038"]

for spring_id in spring_ids:
    print(f"Running {MODEL} for {spring_id}...")

    cleanup_torch()

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
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    def train_xlstm_tune(config):
        criterion = nn.L1Loss()

        model = xLSTMForecaster(
            input_size=len(input_cols),
            hidden_size=config["embedding_dim"],
            output_size=len(FORECAST_DAYS),
            dropout=config["dropout"],
            dense_layers=1,
            architecture=config["architecture"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"]
        )

        best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=config["epochs"],
            patience=3
        )

        tune.report(val_loss=best_val_loss)

    print("     Training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    search_space = {
        "embedding_dim": tune.grid_search([64, 96, 128]),
        "dropout": tune.grid_search([0.1, 0.2]),
        "lr": tune.grid_search([1e-2, 1e-3, 1e-4]),
        "architecture": tune.grid_search([
            "slstm_first",
            "slstm_second",
            "only_slstm",
            "only_mlstm"
        ]),
        "epochs": 100
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=3
    )

    tuner = tune.Tuner(
        train_xlstm_tune,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=72
        ),
        param_space=search_space
    )

    # torch.autograd.set_detect_anomaly(True)

    hyperparam_results = tuner.fit()

    best_result = hyperparam_results.get_best_result(
        metric="val_loss",
        mode="min"
    )
    best_config = best_result.config
    print(best_config)

    tracker = EmissionsTracker(log_level="error")
    tracker.start()
    
    model = xLSTMForecaster(
        input_size=len(input_cols),
        hidden_size=best_config["embedding_dim"],
        output_size=len(FORECAST_DAYS),
        dropout=best_config["dropout"],
        dense_layers=1,
        architecture=best_config["architecture"]
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["lr"]
    )

    MODEL_PATH = os.path.join(
        SPRING_MODEL_DIR,
        f"{MODEL}_{spring_id}_{experiment_timestamp}.pt"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        model_save_path=MODEL_PATH
    )

    emissions_train = tracker.stop()
    energy_kwh_train = tracker.final_emissions_data.energy_consumed

    del train_loader, valid_loader
    del X_train, X_valid, y_train, y_valid
    del optimizer
    del criterion
    cleanup_torch()

    print("     Inference...")
    tracker = EmissionsTracker(log_level="error")
    tracker.start()


    model.eval()   

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=8,
        shuffle=False
    )

    preds = []

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())

    y_pred = torch.cat(preds).numpy()

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
            "embedding_dim": best_config["embedding_dim"],
            "dropout": best_config["dropout"],
            "learning_rate": best_config["lr"],
            "architecture": best_config["architecture"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILEPATH, index=False)

    del model
    del X_test
    del y_test
    del y_pred
    cleanup_torch()
