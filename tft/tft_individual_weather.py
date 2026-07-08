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
from util.sequencing import create_sequences_full_horizon_future

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tft.tft_custom_dataset_weather import TFTCustomDatasetWeather
from tft.tft_train import train_tft

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import TorchNormalizer

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from codecarbon import EmissionsTracker

def cleanup_torch():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

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

MODEL = "TFT_WEATHER"
BATCH_SIZE = 24

MODELS_DIR = os.path.join("tft", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

with open(SPRING_LIST_FILE, 'r') as f:
   spring_ids = [line.strip() for line in f if line.strip()]

# for dev purposes
# spring_ids = ["395038"]

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
    future_known_input_cols = [ "sh", "rr", "tl"]
    known_indices = [input_cols.index(c) for c in future_known_input_cols]

    X_train, y_train, future_train, ts_train  = create_sequences_full_horizon_future(train_df[input_cols], 
                                        train_df[TARGET_COL],
                                        train_df["timestamp"], 
                                        WINDOW_LEN, 
                                        96,
                                        future_known_input_cols)

    X_valid, y_valid, future_valid, ts_valid  = create_sequences_full_horizon_future(valid_df[input_cols], 
                                        valid_df[TARGET_COL],
                                        valid_df["timestamp"], 
                                        WINDOW_LEN, 
                                        96,
                                        future_known_input_cols)

    X_test, y_test, future_test, ts_test  = create_sequences_full_horizon_future(test_df[input_cols], 
                                        test_df[TARGET_COL],
                                        test_df["timestamp"], 
                                        WINDOW_LEN, 
                                        96,
                                        future_known_input_cols)

    train_loader = DataLoader(
        TFTCustomDatasetWeather(X_train, y_train, future_train, known_indices),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        TFTCustomDatasetWeather(X_valid, y_valid, future_valid, known_indices),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        TFTCustomDatasetWeather(X_test, y_test, future_test, known_indices),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    num_input_features = len(input_cols)
    unknown_reals = [c for c in input_cols if c not in future_known_input_cols]
    known_reals = future_known_input_cols

    print("Tuning...")

    config = {
        "hidden_size": tune.grid_search([64, 96]),
        "dropout": 0.1,
        "lr": tune.grid_search([0.001, 0.0001]),
        "lstm_layers": 2,
        "attention_head_size": 4,
        "epochs": 10,
        "patience": 3,
    }

    def trainable_tft(config, X_t=None, y_t=None, future_t=None, X_v=None, y_v=None, future_v=None, known_idx=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(
            TFTCustomDatasetWeather(X_t, y_t, future_t, known_idx),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        valid_loader = DataLoader(
            TFTCustomDatasetWeather(X_v, y_v, future_v, known_idx),
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
            X_t=X_train,
            y_t=y_train,
            future_t=future_train,
            X_v=X_valid,
            y_v=y_valid,
            future_v=future_valid,
            known_idx=known_indices
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
    MODEL_PATH = os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.pt")

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
    
    print("Inference...")

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

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
