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

from codecarbon import EmissionsTracker
from torchinfo import summary

# ---------------------------------------------------------
# SETUP & REPRODUCIBILITY
# ---------------------------------------------------------
SEED = 12019844
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL = "xLSTM_TRANSFER"
BATCH_SIZE = 24
TRANSFER_LR = 1e-3
TRANSFER_EPOCHS = 100
TRANSFER_PATIENCE = 5

MODELS_DIR = os.path.join("xlstm", "models")
LARGE_MODELS_DIR = os.path.join(MODELS_DIR, "large")
LARGE_MODEL_FILENAME = "xLSTM_LARGE_20260612_065528.pt"
LARGE_MODEL_PATH = os.path.join(LARGE_MODELS_DIR, LARGE_MODEL_FILENAME)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILENAME = f"{MODEL}_results_{experiment_timestamp}.csv"
RESULTS_FILEPATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
results = []

# ---------------------------------------------------------
# LARGE MODEL CONFIGURATION (Must match the pre-trained model)
# ---------------------------------------------------------
LM_EMBEDDING_DIM = 64  
LM_DROPOUT = 0.1
LM_ARCHITECTURE = "slstm_first"

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
with open(SPRING_LIST_FILE_TRANSFER, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

def cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

# ---------------------------------------------------------
# TRANSFER LEARNING LOOP
# ---------------------------------------------------------
for spring_id in spring_ids:
    print(f"\n{'='*50}")
    print(f"Running {MODEL} fine-tuning for {spring_id}...")
    print(f"{'='*50}")

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
    
    X_train, y_train, ts_train = create_sequences(train_df[input_cols], train_df[TARGET_COL], train_df["timestamp"], WINDOW_LEN, FORECAST_HS)
    X_valid, y_valid, ts_valid = create_sequences(valid_df[input_cols], valid_df[TARGET_COL], valid_df["timestamp"], WINDOW_LEN, FORECAST_HS)
    X_test, y_test, ts_test = create_sequences(test_df[input_cols], test_df[TARGET_COL], test_df["timestamp"], WINDOW_LEN, FORECAST_HS)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print("Setting up model and freezing layers...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate the model with the large model's architecture
    model = xLSTMForecaster(
        input_size=len(input_cols),
        hidden_size=LM_EMBEDDING_DIM,
        output_size=len(FORECAST_DAYS),
        dropout=LM_DROPOUT,
        architecture=LM_ARCHITECTURE
    ).to(device)

    # 2. Load the pre-trained weights
    try:
        model.load_state_dict(torch.load(LARGE_MODEL_PATH, map_location=device))
        print(f"Successfully loaded large model weights from {LARGE_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading large model weights: {e}")
        sys.exit(1)

    # 3. Print Model Summary (Only print once for the first spring to avoid log clutter)
    if spring_id == spring_ids[0]:
        print("\n--- Pre-trained Model Summary ---")
        summary(model, input_size=(BATCH_SIZE, WINDOW_LEN, len(input_cols)))
        print("---------------------------------\n")

    # 4. Freeze specific layers (input projection & backbone)
    for param in model.input_proj.parameters():
        param.requires_grad = False
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Verify what is unfrozen (should just be dense_stack and output_layer)
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen parameters: {frozen_params:,} | Trainable parameters (Dense & Output): {unfrozen_params:,}")

    # 5. Define optimizer to ONLY update unfrozen parameters
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=TRANSFER_LR
    )

    MODEL_SAVE_PATH = os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.pt")

    print(f"Training (Fine-tuning) {spring_id}...")
    tracker = EmissionsTracker(log_level="error")
    tracker.start()

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=TRANSFER_EPOCHS,
        patience=TRANSFER_PATIENCE,
        model_save_path=MODEL_SAVE_PATH,
        use_ray=False,      # Ray disabled for simple transfer learning
        verbose=True
    )

    emissions_train = tracker.stop()
    energy_kwh_train = tracker.final_emissions_data.energy_consumed
    
    print(f"Inference {spring_id}...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
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

    # Inverse transform
    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred) * 0.001

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
        plt.title(f"{MODEL} (Fine-Tuned) {spring_id} – {d}-Day Ahead Forecast")
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
            "embedding_dim": LM_EMBEDDING_DIM,
            "dropout": LM_DROPOUT,
            "learning_rate": TRANSFER_LR,
            "architecture": LM_ARCHITECTURE,
            "frozen_layers": "input_proj, backbone"
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILEPATH, index=False)

    cleanup_torch()
