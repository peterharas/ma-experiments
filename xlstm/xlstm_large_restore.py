import os
from ray import tune

# Point this to your specific experiment folder
experiment_path = os.path.expanduser("~/ray_results/train_xlstm_2026-06-12_06-59-35")

# 1. Create a totally empty fake function to satisfy Ray's API
def dummy_trainable(config):
    pass

print("Restoring Tuner state from tuner.pkl...")

# 2. Pass the dummy function as the missing positional argument!
tuner = tune.Tuner.restore(experiment_path, dummy_trainable)

# 3. Extract the results
results = tuner.get_results()
best_result = results.get_best_result(metric="val_loss", mode="min")

print("\n--- RECOVERED BEST CONFIG ---")
print(f"Best Val Loss: {best_result.metrics['val_loss']}")
print(best_result.config)
print("-----------------------------\n")