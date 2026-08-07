import pandas as pd
import scipy.stats as stats

# 1. Load the datasets
df_tft = pd.read_csv('results/TFT_results_20260630_092241.csv')
df_weather = pd.read_csv('results/TFT_WEATHER_results_20260710_122632.csv')

# 2. Extract only necessary columns to avoid clutter
df_tft_sub = df_tft[['spring_id', 'horizon', 'nse']]
df_weather_sub = df_weather[['spring_id', 'horizon', 'nse']]

# 3. Merge the datasets on spring_id and horizon
# Using an inner merge guarantees we only compare paired observations 
# (i.e., springs that exist in both datasets for the same horizon)
df_merged = pd.merge(
    df_tft_sub, 
    df_weather_sub, 
    on=['spring_id', 'horizon'], 
    suffixes=('_nofuture', '_futureweather')
)

# 4. Iterate through horizons {1, 2, 3, 4} and perform tests
horizons = [1, 2, 3, 4]

SIGNIFICANCE_LEVEL = 0.05 / len(horizons)

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
    nse_tft = df_h['nse_nofuture']
    nse_weather = df_h['nse_futureweather']
    
    # Calculate the differences between pairs
    differences = nse_tft - nse_weather
    
    # --- ASSUMPTION CHECK: Normality of Differences ---
    # Null Hypothesis: The differences are normally distributed
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    
    print("--- Assumption Check: Shapiro-Wilk Test for Normality ---")
    print(f"Test Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.10f}")
    
    if shapiro_p > SIGNIFICANCE_LEVEL:
        print("Result: Assume Normal Distribution (Fail to reject H0).")
        print("Action: Proceeding with Paired t-test.\n")
        
        # --- PAIRED T-TEST ---
        # Null Hypothesis: The mean difference between paired observations is zero
        t_stat, t_p = stats.ttest_rel(nse_tft, nse_weather)
        
        print("--- Paired T-Test Results ---")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value:     {t_p:.10f}")
        
    else:
        print("Result: Differences are NOT Normally Distributed (Reject H0).")
        print("Action: Assumption violated. Falling back to Wilcoxon signed-rank test.\n")
        
        # --- WILCOXON SIGNED-RANK TEST (Non-parametric) ---
        # Null Hypothesis: The median difference between pairs is zero
        w_stat, w_p = stats.wilcoxon(nse_tft, nse_weather)
        
        print("--- Wilcoxon Signed-Rank Test Results ---")
        print(f"w-statistic: {w_stat:.4f}")
        print(f"p-value:     {w_p:.10f}")

    # Determine significance
    final_p = t_p if shapiro_p > SIGNIFICANCE_LEVEL else w_p
    if final_p < SIGNIFICANCE_LEVEL:
        print("\nConclusion: There is a STATISTICALLY SIGNIFICANT difference between TFT with and without future meteorological inputs.")
    else:
        print("\nConclusion: NO statistically significant difference between the models.")