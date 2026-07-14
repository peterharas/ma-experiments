import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

colors = {
    'LSTM': '#56B4E9',
    'xLSTM': '#0072B2',
    'TRANSFORMER': '#E69F00',
    'TFT': '#D55E00',
    'TFT_WEATHER': '#BB0000',  
}

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Subplot title size (default is 12)
    'axes.labelsize': 14,     # X/Y label size (default is 10)
    'xtick.labelsize': 14,    # X tick label size (default is 10)
    'ytick.labelsize': 14,    # Y tick label size (default is 10)
    'legend.fontsize': 14     # Legend font size (default is 10)
})

df_lstm = pd.read_csv('results/LSTM_ENERGY_results_20260708_190508.csv')
df_xlstm = pd.read_csv('results/xLSTM_results_20260608_090620.csv')
df_transformer = pd.read_csv('results/TRANSFORMER_ENERGY_results_20260709_200314.csv')
df_tft = pd.read_csv('results/TFT_results_20260630_092241.csv')
df_tft_weather = pd.read_csv('results/TFT_WEATHER_results_20260710_122632.csv')

df = pd.concat([df_lstm, df_xlstm, df_transformer, df_tft, df_tft_weather], ignore_index=True)
df['model'] = df['model'].str.replace('_ENERGY', '')
df['model'] = pd.Categorical(df['model'], categories=['LSTM', 'xLSTM', 'TRANSFORMER', 'TFT', 'TFT_WEATHER'], ordered=True)
df = df[df['horizon'] == 1]

df['energy inference [kWh]'] = df['energy inference [kWh]'] * 1000
df['emissions inference [kg CO₂]'] = df['emissions inference [kg CO₂]'] * 1000

rename_mapping = {
    'energy training [kWh]': 'Energy Training [kWh]',
    'energy inference [kWh]': 'Energy Inference [Wh]',
    'emissions training [kg CO₂]': 'Emissions Training [kg CO₂]',
    'emissions inference [kg CO₂]': 'Emissions Inference [g CO₂]'
}
df = df.rename(columns=rename_mapping)

energy_cols = ['Energy Training [kWh]', 'Energy Inference [Wh]']
co2_cols = ['Emissions Training [kg CO₂]', 'Emissions Inference [g CO₂]']

def print_summary(cols, label):
    summary = df.groupby(['model'])[cols].agg(['mean', 'median', 'std']).reset_index()
    print(f"Summary Statistics {label}:")
    print(summary.to_string())
    print(summary.to_latex(index=False, float_format="%.3f"))

print_summary(energy_cols, "Energy")
print_summary(co2_cols, "CO2")

def draw_boxplot(cols, filelabel):
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))

    for i, metric in enumerate(cols):
        sns.violinplot(
            data=df, 
            x='model', 
            y=metric, 
            ax=axes[i], 
            palette=colors
        )
        axes[i].set_xlabel('')
        axes[i].set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'energy', 'plots', f'{filelabel}_individual_violin.png'), dpi=300, bbox_inches='tight')
    plt.show()

draw_boxplot(energy_cols, 'energy')
draw_boxplot(co2_cols, 'co2')