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

SEED = 12019844
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "xLSTM_POC"

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

    
    print("     Training...")
    
    # Detect if CUDA is actually available and working
    # torch.cuda.is_available() can return True even if CUDA is not properly initialized
    cuda_available = False
    if torch.cuda.is_available():
        try:
            # Try to allocate a small tensor on CUDA to verify it works
            torch.zeros(1, device='cuda')
            cuda_available = True
        except RuntimeError:
            cuda_available = False
            print("    Warning: CUDA detected but not functional, falling back to CPU")
    
    device = torch.device("cuda" if cuda_available else "cpu")

    model = xLSTMForecaster(
        input_size=len(input_cols),
        hidden_size=LSTM_UNITS,
        output_size=len(FORECAST_DAYS)
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
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


    print("     Inference...")

    model.load_state_dict(
        torch.load(MODEL_PATH)
    )

    model.eval()   

    with torch.no_grad():

        y_pred = model(
            X_test.to(device)
        )

    y_pred = y_pred.cpu().numpy()

    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)

    # Inverse transform to original scale and convert to m^3/s
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

    for i, d in enumerate(FORECAST_DAYS):
        y_target_d = y_test_orig[:, i]
        y_pred_d = y_pred_orig[:, i]
        metrics = evaluate_forecast(y_target_d, y_pred_d)
        print(f"##### {i} #####")
        print(metrics)
