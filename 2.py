import xarray as xr
import pandas as pd
import scipy.stats as stats
import os
import numpy as np

# 1. Load Data and Extract Reference Cell
data_folder = r"D:\IITK\Sem 1\CE610\P2\Rdata"
ds = xr.open_mfdataset(os.path.join(data_folder, "*.nc"), combine='by_coords')

# Coordinates for Roll No. 261030097
lat_ref = 23.75
lon_ref = 74.25
cell_data = ds.sel(lat=lat_ref, lon=lon_ref, method='nearest')

df = cell_data.to_dataframe().reset_index()
df = df[['time', 'rain']].set_index('time')

# 2. Assign Water Year (June 1 to May 31)
df['WaterYear'] = df.index.year
df.loc[df.index.month < 6, 'WaterYear'] = df.index.year - 1

# Drop incomplete water years at the ends of the record (1950/1951 and potentially 2025/2026)
valid_years = df.groupby('WaterYear').filter(lambda x: len(x) >= 365)
df = valid_years

# 3. Calculate Time Series Metrics
annual_total = df.groupby('WaterYear')['rain'].sum()
annual_max = df.groupby('WaterYear')['rain'].max()

monsoon_data = df[df.index.month.isin([6, 7, 8, 9])]
monsoon_total = monsoon_data.groupby('WaterYear')['rain'].sum()

# 4. Trend Analysis Function (5% Significance Level)
def analyze_trend(series, name):
    x = series.index.values
    y = series.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    is_significant = p_value < 0.05
    trend_dir = "Increasing" if slope > 0 else "Decreasing"
    
    print(f"--- {name} ---")
    print(f"Slope: {slope:.4f} mm/year | p-value: {p_value:.4f}")
    print(f"Statistically Significant (5%): {'Yes' if is_significant else 'No'} ({trend_dir})\n")

print("TREND ANALYSIS RESULTS:")
analyze_trend(annual_total, "Annual Total Rainfall")
analyze_trend(annual_max, "Annual Maximum Daily Rainfall")
analyze_trend(monsoon_total, "Monsoon Total Rainfall")

# 5. IMD Seasons Tabulation
# Define seasons based on months
conditions = [
    df.index.month.isin([1, 2]),
    df.index.month.isin([3, 4, 5]),
    df.index.month.isin([6, 7, 8, 9]),
    df.index.month.isin([10, 11, 12])
]
choices = ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']
df['Season'] = np.select(conditions, choices)

# Calculate monthly averages over the entire period first
monthly_avg = df.groupby(df.index.month)['rain'].mean()
mean_annual_rainfall = monthly_avg.sum()

# Aggregate by Season and Water Year to get seasonal totals per year
seasonal_totals_per_year = df.groupby(['WaterYear', 'Season'])['rain'].sum().unstack()

print("IMD SEASONS SUMMARY:")
print(f"Mean Annual Rainfall: {mean_annual_rainfall:.2f} mm\n")

for season in choices:
    season_data = seasonal_totals_per_year[season].dropna()
    mean_val = season_data.mean()
    std_val = season_data.std()
    cv = std_val / mean_val if mean_val > 0 else 0
    percent_contrib = (mean_val / mean_annual_rainfall) * 100
    
    print(f"Season: {season}")
    print(f"  Mean: {mean_val:.2f} mm")
    print(f"  Standard Deviation: {std_val:.2f} mm")
    print(f"  Coefficient of Variation (CV): {cv:.4f}")
    print(f"  % Contribution: {percent_contrib:.2f}%\n")