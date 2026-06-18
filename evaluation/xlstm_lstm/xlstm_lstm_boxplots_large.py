import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

my_colors = {
    'LSTM': '#56B4E9',   # Sky Blue
    'xLSTM': '#0072B2',  # Deep Marine    
}

# 1. Load the datasets
df_xlstm = pd.read_csv('results/xLSTM_LARGE_results_20260612_065528.csv')
df_lstm = pd.read_csv('results/LSTM_LARGE_results_20260504_063436.csv')

# 2. Combine the datasets
df_combined = pd.concat([df_xlstm, df_lstm], ignore_index=True)

df_combined['model'] = df_combined['model'].str.replace('_LARGE', '')

df_combined['model'] = pd.Categorical(df_combined['model'], categories=['xLSTM', 'LSTM'], ordered=True)

# 3. Define the metrics we want to analyze
metrics = ['nse', 'mae', 'rmse', 'smape']

# 4. Calculate Summary Statistics
summary_stats = df_combined.groupby(['horizon', 'model'])[metrics].agg(['mean', 'median', 'std']).reset_index()

print("Summary Statistics:")
print(summary_stats.to_string())
print(summary_stats.to_latex(index=False, float_format="%.3f"))

# 5. Create Boxplots
fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))

axes = axes.flatten()

for i, metric in enumerate(metrics):
    sns.boxplot(
        data=df_combined, 
        x='horizon', 
        y=metric, 
        hue='model', 
        ax=axes[i], 
        palette=my_colors,
        gap=0.15
    )
    
    # The subplot title line has been removed. 
    # The y-axis label below is sufficient to identify the metric for each subplot.
    axes[i].set_xlabel('Forecast horizon in days', fontsize=12)
    axes[i].set_ylabel(metric.upper(), fontsize=12)
    
    # Increase the size of the tick numbers on both axes
    axes[i].tick_params(axis='both', which='major', labelsize=11)
    
    # Force the legend to the upper right corner and increase its font sizes
    axes[i].legend(loc='upper right', fontsize=11, title_fontsize=12)

# Adjust spacing so titles and labels don't overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save and show the plot (dpi=300 is perfect for high-quality A4 printing)
plt.savefig(os.path.join('evaluation', 'xlstm_lstm_boxplots_large.png'), dpi=300, bbox_inches='tight')
plt.show()