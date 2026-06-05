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

MODEL = "xLSTM_POC"
BATCH_SIZE = 24

MODELS_DIR = os.path.join("xlstm", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

spring_id = "395038"

SPRING_DIR = os.path.join(SRINGS_BASE_DIR, spring_id)
TRAIN_PATH = os.path.join(SPRING_DIR, f"{spring_id}_train.csv")
VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")
TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")

if not os.path.exists(TRAIN_PATH) or not os.path.exists(VALID_PATH) or not os.path.exists(TEST_PATH) or not os.path.exists(SCALER_Y_PATH):
    print(f"Skipping {spring_id} because of missing data")
    sys.exit()

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


print("Warming up xLSTM CUDA build...")

model = xLSTMForecaster(
    input_size=len(input_cols),
    hidden_size=128,
    output_size=len(FORECAST_DAYS),
    dropout=0.1,
    architecture="slstm_first"
).to("cuda")

dummy = torch.randn(2, WINDOW_LEN, len(input_cols), device="cuda")
model(dummy)

print("Warmup done")
del model
torch.cuda.empty_cache()

print("Tuning...")

config = {
    "embedding_dim": tune.grid_search([64, 96, 128]),
    "dropout": tune.grid_search([0.1, 0.2]),
    "lr": tune.grid_search([1e-2, 1e-3, 1e-4]),
    "architecture": "slstm_first",
    "epochs": 100,
    "patience": 3
}

def train(config, train_loader=None, valid_loader=None):

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    device = torch.device("cuda:0")

    model = xLSTMForecaster(
        input_size=len(input_cols),
        hidden_size=config["embedding_dim"],
        output_size=len(FORECAST_DAYS),
        dropout=config["dropout"],
        architecture=config["architecture"]
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        patience=config["patience"]
    )

    tune.report(val_loss=best_val_loss)


scheduler = ASHAScheduler(
    max_t=config["epochs"],
    grace_period=config["patience"],
    reduction_factor=2,
)

trainable = tune.with_resources(
    tune.with_parameters(
        train,
        train_loader=train_loader,
        valid_loader=valid_loader,
    ),
    resources={"gpu": 1}
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

# print("Training...")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
# model = xLSTMForecaster(
#     input_size=len(input_cols),
#     hidden_size=EMBEDDING_SIZE,
#     output_size=len(FORECAST_DAYS),
#     dropout=DROPOUT
# ).to(device)
# 
# criterion = nn.L1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# MODEL_PATH = os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.pt")
# 
# train_model(
#     model=model,
#     train_loader=train_loader,
#     valid_loader=valid_loader,
#     criterion=criterion,
#     optimizer=optimizer,
#     device=device,
#     epochs=EPOCHS,
#     patience=EARLY_STOPPING_PATIENCE,
#     model_save_path=MODEL_PATH
# )
# 
# 
# print("Inference...")
# 
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()   
# 
# preds = []
# with torch.no_grad():
#     for xb, _ in test_loader:
#         xb = xb.to(device)
#         preds.append(model(xb).cpu())
# y_pred = torch.cat(preds).numpy()
# 
# # Inverse transform to original scale and convert to m^3/s
# with open(SCALER_Y_PATH, 'rb') as f:
#     y_scaler = pickle.load(f)
# y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
# y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001
# 
# for i, d in enumerate(FORECAST_DAYS):
#     y_target_d = y_test_orig[:, i]
#     y_pred_d = y_pred_orig[:, i]
#     metrics = evaluate_forecast(y_target_d, y_pred_d)
#     print(f"##### {i} #####")
#     print(metrics)
