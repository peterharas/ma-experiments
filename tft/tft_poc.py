import pandas as pd
import sys
import random
import os

from datetime import datetime

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tft.configuration import KarstSpringConfig
from tft.model import TemporalFusionTransformer
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


SEED = 12019844
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL = "TFT_POC"
BATCH_SIZE = 24
LR = 0.001


MODELS_DIR = os.path.join("xlstm", "models")
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

print("Creating sequences...")
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


print("Training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalFusionTransformer(config=KarstSpringConfig)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
MODEL_PATH = os.path.join(SPRING_MODEL_DIR, f"{MODEL}_{spring_id}_{experiment_timestamp}.pt")

