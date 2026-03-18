
import os
import sys
import pickle
import pandas as pd
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *

SEED = 12019844
np.random.seed(SEED)
random.seed(SEED)

TRAIN_PATH = 'data/395012_train.csv'
VALID_PATH = 'data/395012_valid.csv'
TEST_PATH = 'data/395012_test.csv'

SCALER_Y_PATH = 'data/395012_scale_y.pkl'

TARGET_COL = 'discharge'

FORECAST_DAYS = [1, 2, 3, 4]
FORECAST_15MS = [d * 24 * 4 for d in FORECAST_DAYS]  # 15-min steps

print("Reading data...")
train_df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
valid_df = pd.read_csv(VALID_PATH, parse_dates=['timestamp'])
test_df  = pd.read_csv(TEST_PATH,  parse_dates=['timestamp'])

# Combine train + validation for VAR fitting (common practice)
train_full = pd.concat([train_df, valid_df]).reset_index(drop=True)

input_cols = [c for c in train_df.columns if c != 'timestamp']

print("Fitting VAR model...")
model = VAR(train_full[input_cols])

print("Select lag order automatically (can be tuned)...")
# Select lag order automatically (can be tuned)
# 4 days = 96h = 384 15mins
# lag_order_results = model.select_order(maxlags=384)
# selected_lag = lag_order_results.aic
selected_lag = 24 # (6h)
print(f"Selected lag (AIC): {selected_lag}")

print("Executing model.fit() ...")
var_model = model.fit(selected_lag)

print("Forecasting...")

# We forecast rolling through the test set
history = train_full[input_cols].values.tolist()
predictions = []

max_h = max(FORECAST_15MS)

for i in range(len(test_df) - max_h):
    input_data = np.array(history[-selected_lag:])
    forecast = var_model.forecast(y=input_data, steps=max_h)

    # extract only required horizons
    pred = [forecast[h-1][input_cols.index(TARGET_COL)] for h in FORECAST_15MS]
    predictions.append(pred)

    # append true observation to history
    history.append(test_df[input_cols].iloc[i].values)

y_pred = np.array(predictions)

# Build y_test aligned with predictions
y_test = []
for i in range(len(test_df) - max_h):
    targets = [test_df[TARGET_COL].iloc[i + h - 1] for h in FORECAST_15MS]
    y_test.append(targets)

y_test = np.array(y_test)

with open(SCALER_Y_PATH, 'rb') as f:
    y_scaler = pickle.load(f)

# Inverse scaling
y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
y_pred_orig = y_scaler.inverse_transform(y_pred) * 0.001

timestamps = test_df["timestamp"].iloc[max_h-1:].values[:len(y_pred_orig)]

# Evaluation
for i, d in enumerate(FORECAST_DAYS):
    print(f"\n=== {d}-Day Ahead ===")
    y_target_d = y_test_orig[:, i]
    y_pred_d = y_pred_orig[:, i]
    print(evaluate_forecast(y_target_d, y_pred_d))

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, y_target_d, label="Observed", linewidth=1)
    plt.plot(timestamps, y_pred_d, label="Predicted", linewidth=1)

    plt.title(f"395012 – {d}-Day Ahead Forecast (VAR)")
    plt.xlabel("Time")
    plt.ylabel("Discharge [m³/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Done")
