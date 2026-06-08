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
from util.sequencing import create_sequences

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import xLSTMForecaster
from train import train_model

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from codecarbon import EmissionsTracker

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

MODEL = "xLSTM"
BATCH_SIZE = 24

MODELS_DIR = os.path.join("xlstm", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

# for dev purposes
# spring_ids = ["395038"]

spring_ids.remove("395012")
spring_ids.remove("395038")
spring_ids.remove("395053")
spring_ids.remove("395079")
spring_ids.remove("395103")
spring_ids.remove("395111")
spring_ids.remove("395137")
spring_ids.remove("395145")
spring_ids.remove("395210")
spring_ids.remove("395244")
spring_ids.remove("395251")
spring_ids.remove("395285")


for spring_id in spring_ids:
    print(f"Running {MODEL} for {spring_id}...")

    SPRING_DIR = os.path.join(SPRINGS_BASE_DIR, spring_id)
    TRAIN_PATH = os.path.join(SPRING_DIR, f"{spring_id}_train.csv")
    VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")
    TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
    SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VALID_PATH) or not os.path.exists(TEST_PATH) or not os.path.exists(SCALER_Y_PATH):
        print(f"Skipping {spring_id} because of missing data")
        continue

    SPRING_MODEL_DIR = os.path.join(MODELS_DIR, spring_id)
    os.makedirs(SPRING_MODEL_DIR, exist_ok=True)

    print("Preparing data...")

    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])

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

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Tuning...")

    config = {
        "embedding_dim": tune.grid_search([64, 96, 128]),
        "dropout": 0.1,
        "lr": tune.grid_search([1e-3, 1e-4]),
        "architecture": tune.grid_search([
            "slstm_first",
            "slstm_second",
            "only_slstm",
            "only_mlstm"
        ]),    
        "epochs": 100,
        "patience": 3,
        "input_size": len(input_cols),
        "output_size": len(FORECAST_DAYS)
    }

    def cleanup_torch():
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

    def train_xlstm(config, X_t=None, y_t=None, X_v=None, y_v=None):

        print("CUDA available:", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

        device = torch.device("cuda:0")

        # 2. Build the DataLoaders LOCALLY inside the worker
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        
        valid_loader = DataLoader(
            TensorDataset(X_v, y_v),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = xLSTMForecaster(
            input_size=config["input_size"],
            hidden_size=config["embedding_dim"],
            output_size=config["output_size"],
            dropout=config["dropout"],
            architecture=config["architecture"]
        ).to(device)

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        train_model(
            model=model,
            train_loader=train_loader, # Pass the locally built loaders
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
            train_xlstm,
            X_t=X_train,
            y_t=y_train,
            X_v=X_valid,
            y_v=y_valid,
        ),
        resources={"cpu": 4, "gpu": 1}
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            max_concurrent_trials=1,  # important for debugging
        ),
        param_space=config,
    )
    tuning_results = tuner.fit()
    best_result = tuning_results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    print(best_config)

    cleanup_torch()

    print("Training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    model = xLSTMForecaster(
        input_size=best_config["input_size"],
        hidden_size=best_config["embedding_dim"],
        output_size=best_config["output_size"],
        dropout=best_config["dropout"],
        architecture=best_config["architecture"]
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])
    MODEL_PATH = os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.pt")

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        patience=config["patience"],
        model_save_path=MODEL_PATH
    )

    emissions_train = tracker.stop()
    energy_kwh_train = tracker.final_emissions_data.energy_consumed
    
    print("Inference...")

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
    y_pred = torch.cat(preds).numpy()

    emissions_inference = tracker.stop()
    energy_kwh_inference = tracker.final_emissions_data.energy_consumed

    # Inverse transform to original scale and convert to m^3/s
    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)
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

    cleanup_torch()
