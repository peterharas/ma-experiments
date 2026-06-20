import os
import pandas as pd


# 1. Load the datasets
df = pd.read_csv('results/BASELINE_LASTKNOWN_results_20260427_080725.csv')

# 2. Define the metrics we want to analyze
metrics = ['nse', 'mae', 'rmse', 'smape']

# 3. Calculate Summary Statistics
summary_stats = df.groupby(['horizon'])[metrics].agg(['mean', 'median', 'std']).reset_index()

print("Summary Statistics:")
print(summary_stats.to_string())
print(summary_stats.to_latex(index=False, float_format="%.3f"))
