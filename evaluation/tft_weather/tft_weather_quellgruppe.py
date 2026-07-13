import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Subplot title size
    'axes.labelsize': 14,     # X/Y label size
    'xtick.labelsize': 14,    # X tick label size
    'ytick.labelsize': 14,    # Y tick label size
    'legend.fontsize': 14     # Legend font size
})

# Load the datasets
df_tft = pd.read_csv('results/TFT_results_20260630_092241.csv')
df_tft_weather = pd.read_csv('results/TFT_WEATHER_results_20260710_122632.csv')
df_springs_clusters = pd.read_csv('evaluation/springs-clusters.csv')

# Group by spring_id and calculate the mean NSE across the four horizons
mean_nse_tft = df_tft.groupby('spring_id')['nse'].mean().reset_index()
mean_nse_tft_weather = df_tft_weather.groupby('spring_id')['nse'].mean().reset_index()

# Merge the two DataFrames on spring_id and calculate the difference
merged_diff = pd.merge(
    mean_nse_tft_weather,
    mean_nse_tft,
    on='spring_id',
    suffixes=('_weather', '_tft')
)
merged_diff['nse_diff'] = merged_diff['nse_weather'] - merged_diff['nse_tft']

# Merge with the spring-clusters DataFrame on HZBNr. (spring_id)
final_df = pd.merge(
    merged_diff,
    df_springs_clusters,
    left_on='spring_id',
    right_on='HZBNr.',
    how='left'
)

# Count the number of springs in each Quellgruppe
quellgruppe_counts = final_df['Quellgruppe'].value_counts().sort_index()

print("Number of springs per Quellgruppe:")
print(quellgruppe_counts)

# Plot boxplots for the differences by Quellgruppe
plt.figure(figsize=(11.69, 8.27))
sns.boxplot(
    x='Quellgruppe',
    y='nse_diff',
    data=final_df,
    palette='Set2',
    gap=0.15
)

# Add title and labels
plt.title('Paired Differences (TFT_WEATHER − TFT), Mean of all Horizons by Spring Group', fontsize=16)
plt.xlabel('Spring Group', fontsize=14)
plt.ylabel('Δ NSE', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save and show the plot
plt.savefig(os.path.join('evaluation', 'tft_weather', 'plots', 'nse_diff_by_quellgruppe.png'), dpi=300, bbox_inches='tight')
plt.show()