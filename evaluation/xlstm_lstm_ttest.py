import pandas as pd
import numpy as np
import scipy.stats as stats

# 1. Load the datasets
df_xlstm = pd.read_csv('results/xLSTM_results_20260608_090620.csv')
df_lstm = pd.read_csv('results/LSTM_results_20260429_093001.csv')

# 2. Extract only necessary columns to avoid clutter
df_xlstm_sub = df_xlstm[['spring_id', 'horizon', 'nse']]
df_lstm_sub = df_lstm[['spring_id', 'horizon', 'nse']]

# 3. Merge the datasets on spring_id and horizon
# Using an inner merge guarantees we only compare paired observations 
# (i.e., springs that exist in both datasets for the same horizon)
df_merged = pd.merge(
    df_xlstm_sub, 
    df_lstm_sub, 
    on=['spring_id', 'horizon'], 
    suffixes=('_xlstm', '_lstm')
)

# 4. Iterate through horizons {1, 2, 3, 4} and perform tests
horizons = [1, 2, 3, 4]

for h in horizons:
    print(f"\n{'='*40}")
    print(f"HORIZON {h}")
    print(f"{'='*40}")
    
    # Isolate data for the current horizon
    df_h = df_merged[df_merged['horizon'] == h]
    
    if df_h.empty:
        print("No data available for this horizon.")
        continue

    # Extract arrays for testing
    nse_xlstm = df_h['nse_xlstm']
    nse_lstm = df_h['nse_lstm']
    
    # Calculate the differences between pairs
    differences = nse_xlstm - nse_lstm
    
    # --- ASSUMPTION CHECK: Normality of Differences ---
    # Null Hypothesis: The differences are normally distributed
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    
    print("--- Assumption Check: Shapiro-Wilk Test for Normality ---")
    print(f"Test Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.5f}")
    
    if shapiro_p > 0.05:
        print("Result: Assume Normal Distribution (Fail to reject H0).")
        print("Action: Proceeding with Paired t-test.\n")
        
        # --- PAIRED T-TEST ---
        # Null Hypothesis: The mean difference between paired observations is zero
        t_stat, t_p = stats.ttest_rel(nse_xlstm, nse_lstm)
        
        print("--- Paired T-Test Results ---")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value:     {t_p:.5f}")
        
    else:
        print("Result: Differences are NOT Normally Distributed (Reject H0).")
        print("Action: Assumption violated. Falling back to Wilcoxon signed-rank test.\n")
        
        # --- WILCOXON SIGNED-RANK TEST (Non-parametric) ---
        # Null Hypothesis: The median difference between pairs is zero
        w_stat, w_p = stats.wilcoxon(nse_xlstm, nse_lstm)
        
        print("--- Wilcoxon Signed-Rank Test Results ---")
        print(f"w-statistic: {w_stat:.4f}")
        print(f"p-value:     {w_p:.5f}")

    # Determine significance (assuming alpha = 0.05)
    final_p = t_p if shapiro_p > 0.05 else w_p
    if final_p < 0.05:
        print("\nConclusion: There is a STATISTICALLY SIGNIFICANT difference between xLSTM and LSTM.")
    else:
        print("\nConclusion: NO statistically significant difference between the models.")