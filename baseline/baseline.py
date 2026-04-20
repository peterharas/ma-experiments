from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.metrics import *
from util.paths import *
from util.experiment_params import *
from util.sequencing import create_sequences


MODEL = "BASELINE"

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = []

with open(SPRING_LIST_FILE, 'r') as f:
    spring_ids = [line.strip() for line in f if line.strip()]

for spring_id in spring_ids:
    print(f"Calculating baseline for {spring_id}...")

    SPRING_DIR = os.path.join(SRINGS_BASE_DIR, spring_id)
    TEST_PATH = os.path.join(SPRING_DIR, f"{spring_id}_test.csv")
    SCALER_Y_PATH = os.path.join(SPRING_DIR, f"{spring_id}_scale_y.pkl")
    
    if not os.path.exists(TEST_PATH) or not os.path.exists(SCALER_Y_PATH):
        print(f"    Skipping {spring_id} because of missing data")
        continue

    test_df = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])
    input_cols = [c for c in test_df.columns if c not in ['timestamp']]
    X_test, y_test, ts_test  = create_sequences(test_df[input_cols], 
                                       test_df[TARGET_COL],
                                       test_df["timestamp"], 
                                       WINDOW_LEN, 
                                       FORECAST_15MS)

    X_test_target =  X_test[:, :, 1]
    means = X_test_target.mean(axis=1)
    y_pred = np.repeat(means[:, np.newaxis], 4, axis=1)

    with open(SCALER_Y_PATH, 'rb') as f:
        y_scaler = pickle.load(f)

    # Inverse transform to original scale and convert to m^3/s
    y_test_orig = y_scaler.inverse_transform(y_test) * 0.001
    y_pred_orig = y_scaler.inverse_transform(y_pred)  * 0.001

    # Evaluation
    for i, d in enumerate(FORECAST_DAYS):
        # print(f"\n  === {spring_id} {d}-Day Ahead ===")
        y_target_d = y_test_orig[:, i]
        y_pred_d = y_pred_orig[:, i]
        metrics = evaluate_forecast(y_target_d, y_pred_d)
        # print(f"    {metrics}")

        plots_base_dir = os.path.join(RESULTS_PLOTS_DIR, spring_id)
        os.makedirs(plots_base_dir, exist_ok=True)
        plot_filename = f"{spring_id}_{MODEL}_{d}d_{experiment_timestamp}.png"
        plot_path = os.path.join(plots_base_dir, plot_filename)

        timestamps_d = ts_test[:, i]

        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_d, y_test_orig[:, i], label="Observed", linewidth=1)
        plt.plot(timestamps_d, y_pred_orig[:, i], label="Predicted", linewidth=1)
        plt.title(f"{spring_id} – {d}-Day Ahead Forecast")
        plt.xlabel("Time")
        plt.ylabel("Discharge [m³/s]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()  # important to avoid memory issues

        # Save results
        results.append({
            "spring_id": spring_id,
            "model": MODEL,
            "horizon": d,
            "nse": metrics["nse"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "smape": metrics["smape"],
        })

results_df = pd.DataFrame(results)

os.makedirs(RESULTS_DIR, exist_ok=True)

filename = f"{MODEL}_results_{experiment_timestamp}.csv"

save_path = os.path.join(RESULTS_DIR, filename)
results_df.to_csv(save_path, index=False)

print(f"\nResults saved to: {save_path}")