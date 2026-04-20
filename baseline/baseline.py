import os
import sys
import pandas as pd

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences


with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

# For dev purposes
spring_ids = ["395012"]

for spring_id in spring_ids:
    print(f"Calculating baseline for {spring_id}...")

    SPRING_DIR = os.path.join(SRINGS_BASE_DIR, spring_id)
    TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
    SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")

    test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])
    input_cols = [c for c in test_df.columns if c not in ['timestamp']]
    X_test, y_test  = create_sequences(test_df[input_cols], test_df[TARGET_COL], WINDOW_LEN, FORECAST_15MS)

    X_test_target =  X_test[:, :, 1]
    means = X_test_target.mean(axis=1)
    y_pred = np.repeat(means[:, np.newaxis], 4, axis=1)

    print("break")
