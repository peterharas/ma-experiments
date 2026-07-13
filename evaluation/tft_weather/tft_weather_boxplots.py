import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

my_colors = {
    'TFT_WEATHER': '#BB0000',
    'TFT': '#D55E00'
}

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Subplot title size (default is 12)
    'axes.labelsize': 14,     # X/Y label size (default is 10)
    'xtick.labelsize': 14,    # X tick label size (default is 10)
    'ytick.labelsize': 14,    # Y tick label size (default is 10)
    'legend.fontsize': 14     # Legend font size (default is 10)
})

# 1. Load the datasets
df_tft = pd.read_csv('results/TFT_results_20260630_092241.csv')
df_weather = pd.read_csv('results/TFT_WEATHER_results_20260710_122632.csv')

# 2. Combine the datasets
df_combined = pd.concat([df_weather, df_tft], ignore_index=True)
df_combined['model'] = pd.Categorical(df_combined['model'], categories=['TFT_WEATHER', 'TFT'], ordered=True)

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
    axes[i].set_xlabel('Forecast horizon in days')
    axes[i].set_ylabel(metric.upper())
    
    # Increase the size of the tick numbers on both axes
    axes[i].tick_params(axis='both', which='major')
    
    loc = 'upper right' if i > 0 else 'lower right'
    axes[i].legend(loc=loc)

# Adjust spacing so titles and labels don't overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save and show the plot (dpi=300 is perfect for high-quality A4 printing)
plt.savefig(os.path.join('evaluation', 'tft_weather', 'plots', 'tft_weather_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()