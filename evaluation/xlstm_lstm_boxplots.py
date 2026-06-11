import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the datasets
df_xlstm = pd.read_csv('results/xLSTM_results_20260608_090620.csv')
df_lstm = pd.read_csv('results/LSTM_results_20260429_093001.csv')

# 2. Combine the datasets
# We can just concatenate them since they already have a 'model' column distinguishing them
df_combined = pd.concat([df_xlstm, df_lstm], ignore_index=True)

# 3. Define the metrics we want to analyze
metrics = ['nse', 'mae', 'rmse', 'smape']

# 4. Calculate Summary Statistics (mean, median, standard deviation)
# Grouping by 'horizon' and 'model'
summary_stats = df_combined.groupby(['horizon', 'model'])[metrics].agg(['mean', 'median', 'std']).reset_index()

print("Summary Statistics:")
print(summary_stats.to_string())

# Optional: Save the summary statistics to a CSV for your records
# summary_stats.to_csv('summary_statistics.csv', index=False)

# 5. Create Boxplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    # Setting x='horizon' and hue='model' places the LSTM and xLSTM boxes side-by-side for each horizon
    sns.boxplot(
        data=df_combined, 
        x='horizon', 
        y=metric, 
        hue='model', 
        ax=axes[i], 
        palette='Set2',
        gap=0.15
    )
    axes[i].set_title(f'{metric.upper()} by Horizon, individual models (LSTM vs xLSTM)')
    axes[i].set_xlabel('Horizon')
    axes[i].set_ylabel(metric.upper())

plt.tight_layout()

# Save and show the plot
plt.savefig('lstm_vs_xlstm_boxplots.png', dpi=300)
plt.show()