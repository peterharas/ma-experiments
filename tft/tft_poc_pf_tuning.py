import gc
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

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import TorchNormalizer

from tft.tft_custom_dataset import TFTCustomDataset
from tft.tft_train import train_tft

from ray import tune
from ray.tune.schedulers import ASHAScheduler

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

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "TFT_POC"
BATCH_SIZE = 24

MODELS_DIR = os.path.join("tft", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

spring_id = "395038"

SPRING_DIR = os.path.join(SPRINGS_BASE_DIR, spring_id)
TRAIN_PATH = os.path.join(SPRING_DIR, f"{spring_id}_train.csv")
VALID_PATH = os.path.join(SPRING_DIR, f"{spring_id}_valid.csv")
TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")

if not os.path.exists(TRAIN_PATH) or not os.path.exists(VALID_PATH) or not os.path.exists(TEST_PATH) or not os.path.exists(SCALER_Y_PATH):
    print(f"Skipping {spring_id} because of missing data")
    exit

SPRING_MODEL_DIR = os.path.join(MODELS_DIR, spring_id)
os.makedirs(SPRING_MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])

# For dev purposes
# train_df = train_df.iloc[:1680]
# valid_df = valid_df.iloc[:1680]
# test_df = test_df.iloc[:1680]

print("Creating sequences...")
input_cols = [c for c in test_df.columns if c not in ['timestamp']]

X_train, y_train, ts_train  = create_sequences_full_horizon(train_df[input_cols], 
                                    train_df[TARGET_COL],
                                    train_df["timestamp"], 
                                    WINDOW_LEN, 
                                    96)

X_valid, y_valid, ts_valid  = create_sequences_full_horizon(valid_df[input_cols], 
                                    valid_df[TARGET_COL],
                                    valid_df["timestamp"], 
                                    WINDOW_LEN, 
                                    96)

X_test, y_test, ts_test  = create_sequences_full_horizon(test_df[input_cols], 
                                    test_df[TARGET_COL],
                                    test_df["timestamp"], 
                                    WINDOW_LEN, 
                                    96)


num_input_features = len(input_cols)
# Define our variables
unknown_reals = [f"feat_{i}" for i in range(num_input_features)]
known_reals = ["dummy_known"]

print("Tuning...")

config = {
    "hidden_size": tune.grid_search([64, 96, 168]),
    "dropout": 0.1,
    "lr": tune.grid_search([0.001, 0.0001]),
    "lstm_layers": tune.grid_search([1, 2]),
    "attention_head_size": 4,
    "epochs": 10,
    "patience": 3,
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

train_loader = DataLoader(
    TFTCustomDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    TFTCustomDataset(X_valid, y_valid),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# You will likely also need your test_loader later for inference
test_loader = DataLoader(
    TFTCustomDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


print("Inference...")

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()   

preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        for k, v in batch_x.items():
            batch_x[k] = v.to(device)
            
        out = model(batch_x)
        full_preds = out.prediction.squeeze(-1).cpu() # Shape: (Batch, 96)
        
        # Extract indices: 23 (24h), 47 (48h), 71 (72h), 95 (96h)
        horizon_indices = [23, 47, 71, 95]
        target_preds = full_preds[:, horizon_indices] # Shape: (Batch, 4)
        
        preds.append(target_preds)

y_pred = torch.cat(preds).numpy()

# Inverse transform to original scale and convert to m^3/s
with open(SCALER_Y_PATH, 'rb') as f:
    y_scaler = pickle.load(f)
y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

for i, d in enumerate(FORECAST_DAYS):
    y_target_d = y_test_orig[:, i]
    y_pred_d = y_pred_orig[:, i]
    metrics = evaluate_forecast(y_target_d, y_pred_d)
    print(f"##### {i} #####")
    print(metrics)
