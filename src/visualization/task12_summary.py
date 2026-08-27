import pandas as pd
import matplotlib.pyplot as plt

# 1. load and concatenate dataframes
df1 = pd.read_csv('simulation_results_part1.csv')
df2 = pd.read_csv('simulation_results_part2.csv')
full_df = pd.concat([df1, df2], ignore_index=True)

# --- a) Histogram of distribution ---
plt.figure(figsize=(10, 6))
full_df['mean_temp'].hist(bins=40, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Mean Temperatures across 4,571 Buildings', fontsize=14)
plt.xlabel('Mean Temperature (°C)', fontsize=12)
plt.ylabel('Number of Buildings', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('mean_temp_distribution.png', dpi=300) # save the histogram as a PNG file
print("a) Histogram saved as 'mean_temp_distribution.png'")

# --- b) Average mean temperature ---
avg_mean = full_df['mean_temp'].mean()
print(f"b) Average mean temperature: {avg_mean:.2f} °C")

# --- c) Average standard deviation ---
avg_std = full_df['std_temp'].mean()
print(f"c) Average temperature standard deviation: {avg_std:.4f}")

# --- d) Comfortable buildings (At least 50% area > 18ºC) ---
# Using boolean indexing to count rows that meet the condition
count_comfort = (full_df['pct_above_18'] >= 50).sum()
print(f"d) Number of buildings with ≥ 50% area above 18ºC: {count_comfort}")

# --- e) Cold buildings statistics (At least 50% area < 15ºC) ---
count_cold = (full_df['pct_below_15'] >= 50).sum()
print(f"e) Number of buildings with ≥ 50% area below 15ºC: {count_cold}")