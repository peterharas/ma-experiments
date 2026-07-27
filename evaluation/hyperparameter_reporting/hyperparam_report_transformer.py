import pandas as pd

df = pd.read_csv('results/TRANSFORMER_results_20260430_193701.csv')
df_filtered = df[df['horizon'] == 1]
cols_to_keep = ['spring_id', 'head_size', 'num_heads', 'ff_dim', 'num_transformer_blocks', 'mlp_units']
df_table = df_filtered[cols_to_keep]

latex_table = df_table.to_latex(index=False)
print(latex_table)
