import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

colors = {
    'LSTM': '#56B4E9',
    'xLSTM': '#0072B2',
    'TRANSFORMER': '#E69F00',
    'TFT': '#D55E00',
}

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Subplot title size (default is 12)
    'axes.labelsize': 14,     # X/Y label size (default is 10)
    'xtick.labelsize': 14,    # X tick label size (default is 10)
    'ytick.labelsize': 14,    # Y tick label size (default is 10)
    'legend.fontsize': 14     # Legend font size (default is 10)
})

df_lstm = pd.read_csv('results/LSTM_LARGE_ENERGY_results_20260709_103328.csv')
df_xlstm = pd.read_csv('results/xLSTM_LARGE_results_20260612_065528.csv')
df_transformer = pd.read_csv('results/TRANSFORMER_LARGE_ENERGY_results_20260710_075029.csv')
df_tft = pd.read_csv('results/TFT_LARGE_results_20260707_200924.csv')

df = pd.concat([df_lstm, df_xlstm, df_transformer, df_tft], ignore_index=True)
df['model'] = df['model'].str.replace('_LARGE', '')
df['model'] = df['model'].str.replace('_ENERGY', '')
df['model'] = pd.Categorical(df['model'], categories=['LSTM', 'xLSTM', 'TRANSFORMER', 'TFT'], ordered=True)

df['energy inference [kWh]'] = df['energy inference [kWh]'] * 1000
df['emissions inference [kg CO₂]'] = df['emissions inference [kg CO₂]'] * 1000

rename_mapping = {
    'energy training [kWh]': 'Energy Training [kWh]',
    'energy inference [kWh]': 'Energy Inference [Wh]',
    'emissions training [kg CO₂]': 'Emissions Training [kg CO₂]',
    'emissions inference [kg CO₂]': 'Emissions Inference [g CO₂]'
}
df = df.rename(columns=rename_mapping)

df_train = df[(df['horizon'] == 1) & (df['spring_id'] == 395012)]

df_train_print = df_train[['model', 'Energy Training [kWh]', 'Emissions Training [kg CO₂]']]
print("Large model train energy consumption and emissions")
print(df_train_print.to_string())
print(df_train_print.to_latex(index=False, float_format="%.3f"))

train_cols = ['Energy Training [kWh]', 'Emissions Training [kg CO₂]']
fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
for i, metric in enumerate(train_cols):
    sns.barplot(
        data=df_train, 
        x='model', 
        y=metric, 
        ax=axes[i], 
        palette=colors
    )
    axes[i].set_xlabel('')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig(os.path.join('evaluation', 'energy', 'plots', f'energy_co2_train_large_barplot.png'), dpi=300, bbox_inches='tight')
plt.show()


df_inference = df[df['horizon'] == 1]
inference_cols = ['Energy Inference [Wh]', 'Emissions Inference [g CO₂]']

fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
for i, metric in enumerate(inference_cols):
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
plt.savefig(os.path.join('evaluation', 'energy', 'plots', f'energy_co2_inference_large_violin.png'), dpi=300, bbox_inches='tight')
plt.show()

summary = df_inference.groupby(['model'])[inference_cols].agg(['mean', 'median', 'std']).reset_index()
print(f"Summary Statistics Inference:")
print(summary.to_string())
print(summary.to_latex(index=False, float_format="%.3f"))
