import os
from ray import tune

# Point this directly to the folder containing the tuner.pkl
experiment_path = os.path.expanduser("~/ray_results/train_xlstm_2026-06-12_06-59-35")

print("Restoring Tuner state...")
tuner = tune.Tuner.restore(experiment_path)

# Fetch all results from the restored experiment
results = tuner.get_results()

# Extract the best one based on your metric
best_result = results.get_best_result(metric="val_loss", mode="min")

print("\n--- RECOVERED BEST CONFIG ---")
print(f"Best Val Loss: {best_result.metrics['val_loss']}")
print(best_result.config)
print("-----------------------------\n")