import xarray as xr
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 1. DATA LOADING & EXTRACTION (No Dask Required)
# ==========================================
data_folder = r"D:\IITK\Sem 1\CE610\P2\Rdata"

# Reference grid cell for Som River
lat_ref = 23.75
lon_ref = 74.25

print(f"Extracting data for Lat: {lat_ref}, Lon: {lon_ref}...")
file_list = glob.glob(os.path.join(data_folder, "*.nc"))
df_list = []

for file in file_list:
    ds = xr.open_dataset(file)
    
    # Case-insensitive search for coordinate and time names
    all_dims_coords = list(ds.coords) + list(ds.dims)
    lat_name = next(c for c in all_dims_coords if c.lower() in ['lat', 'latitude'])
    lon_name = next(c for c in all_dims_coords if c.lower() in ['lon', 'longitude'])
    time_name = next(c for c in all_dims_coords if c.lower() in ['time', 'date'])
    
    # Identify the rainfall variable (usually the only data variable)
    rain_var = list(ds.data_vars)[0] 
    
    # Extract nearest grid cell
    cell_data = ds.sel({lat_name: lat_ref, lon_name: lon_ref}, method='nearest')
    temp_df = cell_data.to_dataframe().reset_index()
    
    # Standardize column names regardless of what they are called in the file
    temp_df = temp_df.rename(columns={rain_var: 'rain', time_name: 'time'})
    
    # Keep only time and rain columns
    df_list.append(temp_df[['time', 'rain']])
    ds.close()

# Combine all years into a single continuous DataFrame
df = pd.concat(df_list).set_index('time').sort_index()

# Ensure the index is a proper datetime format
df.index = pd.to_datetime(df.index)

# Replace negative fill values with 0
df['rain'] = df['rain'].apply(lambda x: 0 if x < 0 else x)

# ==========================================
# 2. WATER YEAR ASSIGNMENT
# ==========================================
# Water year starts on June 1. If month < 6, it belongs to the previous year.
df['WaterYear'] = np.where(df.index.month < 6, df.index.year - 1, df.index.year)

# Drop incomplete water years at the boundaries (must have 365 or 366 days)
valid_years = df.groupby('WaterYear').filter(lambda x: len(x) >= 365)
df = valid_years

# ==========================================
# 3. TIME SERIES & TREND ANALYSIS
# ==========================================
annual_total = df.groupby('WaterYear')['rain'].sum()
annual_max = df.groupby('WaterYear')['rain'].max()

monsoon_data = df[df.index.month.isin([6, 7, 8, 9])]
monsoon_total = monsoon_data.groupby('WaterYear')['rain'].sum()

def analyze_trend(series, name):
    x = series.index.values
    y = series.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    is_sig = p_value < 0.05
    direction = "increasing" if slope > 0 else "decreasing"
    
    print(f"--- {name} Trend Analysis ---")
    print(f"Slope: {slope:.4f} mm/year | p-value: {p_value:.4f}")
    if is_sig:
        print(f"Comment: The {name.lower()} shows a STATISTICALLY SIGNIFICANT {direction} trend.\n")
    else:
        print(f"Comment: The {name.lower()} shows a {direction} trend, but it is NOT statistically significant.\n")

print("\n==========================================")
print("TREND ANALYSIS RESULTS")
print("==========================================")
analyze_trend(annual_total, "Annual Total Rainfall")
analyze_trend(annual_max, "Annual Maximum Daily Rainfall")
analyze_trend(monsoon_total, "Monsoon Rainfall")

# Plotting Time Series
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
annual_total.plot(ax=axes[0], title='Annual Total Rainfall (Water Year)', ylabel='Rainfall (mm)', color='b')
annual_max.plot(ax=axes[1], title='Annual Maximum Daily Rainfall', ylabel='Rainfall (mm)', color='r')
monsoon_total.plot(ax=axes[2], title='Monsoon Total Rainfall (Jun-Sep)', ylabel='Rainfall (mm)', color='g')
plt.tight_layout()
plt.savefig(os.path.join(data_folder, "TimeSeries_Trends.png"))
plt.show()

# ==========================================
# 4. DDF AND IDF CURVES (Gumbel Distribution)
# ==========================================
durations = [1, 2, 3, 5, 7, 10, 15, 30] # days
return_periods = [2, 5, 10, 25, 50, 100] # years

print("Calculating DDF and IDF Curves...")
annual_max_durations = {}
for D in durations:
    rolling_sum = df['rain'].rolling(window=D, min_periods=1).sum()
    temp_df = pd.DataFrame({'rain_sum': rolling_sum, 'WaterYear': df['WaterYear']})
    annual_max_durations[D] = temp_df.groupby('WaterYear')['rain_sum'].max()

def gumbel_factor(T):
    return -(np.sqrt(6)/np.pi) * (0.5772 + np.log(np.log(T / (T - 1))))

K_factors = {T: gumbel_factor(T) for T in return_periods}

ddf_results = pd.DataFrame(index=durations, columns=return_periods)
idf_results = pd.DataFrame(index=durations, columns=return_periods)

for D in durations:
    series = annual_max_durations[D]
    mean_val = series.mean()
    std_val = series.std()
    
    for T in return_periods:
        depth = mean_val + K_factors[T] * std_val
        ddf_results.loc[D, T] = depth
        idf_results.loc[D, T] = depth / (D * 24) # Intensity in mm/hr

# Plotting DDF and IDF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for T in return_periods:
    ax1.plot(ddf_results.index, ddf_results[T], marker='o', label=f'{T}-Year')
    ax2.plot(idf_results.index, idf_results[T], marker='o', label=f'{T}-Year')

ax1.set(title='Depth-Duration-Frequency (DDF) Curves', xlabel='Duration (Days)', ylabel='Depth (mm)')
ax1.legend(title="Return Period")
ax1.grid(True, which="both", ls="--")

ax2.set(title='Intensity-Duration-Frequency (IDF) Curves', xlabel='Duration (Days)', ylabel='Intensity (mm/hr)')
ax2.legend(title="Return Period")
ax2.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig(os.path.join(data_folder, "DDF_IDF_Curves.png"))
plt.show()

# ==========================================
# 5. MONTHLY AVERAGE & IMD SEASONS
# ==========================================
monthly_avg = df.groupby(df.index.month)['rain'].mean()
mean_annual_rainfall = monthly_avg.sum()

plt.figure(figsize=(8, 4))
monthly_avg.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Monthly Average Rainfall')
plt.xlabel('Month')
plt.ylabel('Average Rainfall (mm)')
plt.xticks(ticks=range(12), labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(data_folder, "Monthly_Average.png"))
plt.show()

conditions = [
    df.index.month.isin([1, 2]),
    df.index.month.isin([3, 4, 5]),
    df.index.month.isin([6, 7, 8, 9]),
    df.index.month.isin([10, 11, 12])
]
choices = ['Winter (Jan-Feb)', 'Pre-Monsoon (Mar-May)', 'Monsoon (Jun-Sep)', 'Post-Monsoon (Oct-Dec)']
df['Season'] = np.select(conditions, choices)

seasonal_totals_per_year = df.groupby(['WaterYear', 'Season'])['rain'].sum().unstack()

print("\n==========================================")
print("IMD SEASONS SUMMARY")
print(f"Mean Annual Rainfall: {mean_annual_rainfall:.2f} mm")
print("==========================================")

for season in choices:
    if season in seasonal_totals_per_year.columns:
        season_data = seasonal_totals_per_year[season].dropna()
        mean_val = season_data.mean()
        std_val = season_data.std()
        cv = std_val / mean_val if mean_val > 0 else 0
        percent_contrib = (mean_val / mean_annual_rainfall) * 100
        
        print(f"{season}:")
        print(f"  Mean:                  {mean_val:.2f} mm")
        print(f"  Standard Deviation:    {std_val:.2f} mm")
        print(f"  Coeff. of Variation:   {cv:.4f}")
        print(f"  % of Annual Rainfall:  {percent_contrib:.2f}%\n")