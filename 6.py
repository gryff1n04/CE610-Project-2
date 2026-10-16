import xarray as xr
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. FILE PATHS & CONSTANTS
# ==========================================
# Update this to the exact path of your downloaded NetCDF file
nc_file_path = r"D:\IITK\Sem 1\CE610\P2\ERA5_data\reanalysis-era5-single-levels-timeseries-sfcsh6kq3z_.nc" 
output_folder = os.path.dirname(nc_file_path)

lat_ref = 23.75
elevation_est = 250 # Adjust if you have a more precise average elevation for your reference grid

print("Loading NetCDF file...")

# ==========================================
# 2. FAO-56 PENMAN-MONTEITH FUNCTION
# ==========================================
def calc_fao56_et0(tmean, tmax, tmin, tdew, wind_2m, rs_mj, elevation, lat_deg, day_of_year):
    """Calculates daily ET0 (mm/day) using FAO-56 standard equations."""
    lat_rad = np.radians(lat_deg)
    
    P = 101.3 * ((293 - 0.0065 * elevation) / 293)**5.26
    gamma = 0.000665 * P
    
    e0_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    e0_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (e0_tmax + e0_tmin) / 2.0
    ea = 0.6108 * np.exp(17.27 * tdew / (tdew + 237.3))
    
    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3)**2
    
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    solar_dec = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(solar_dec))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(solar_dec) + 
        np.cos(lat_rad) * np.cos(solar_dec) * np.sin(ws)
    )
    
    Rso = (0.75 + 2e-5 * elevation) * Ra
    Rns = (1 - 0.23) * rs_mj
    rs_rso_ratio = np.clip(rs_mj / Rso, 0.3, 1.0)
    
    sigma = 4.903e-9
    Rnl = sigma * (((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * rs_rso_ratio - 0.35)
    Rn = Rns - Rnl
    
    num = 0.408 * delta * Rn + gamma * (900 / (tmean + 273)) * wind_2m * (es - ea)
    den = delta + gamma * (1 + 0.34 * wind_2m)
    et0 = num / den
    
    return np.maximum(et0, 0)

# ==========================================
# 3. DATA PROCESSING
# ==========================================
print("Processing hourly variables into daily values...")
ds = xr.open_dataset(nc_file_path)

# Convert directly to dataframe since the file is already a single point
df_hourly = ds.to_dataframe().reset_index()

# Set the time index properly (ERA5 usually uses 'time' or 'valid_time')
time_col = 'valid_time' if 'valid_time' in df_hourly.columns else 'time'
df_hourly = df_hourly.set_index(time_col)

# Convert units
df_hourly['t2m_C'] = df_hourly['t2m'] - 273.15
df_hourly['d2m_C'] = df_hourly['d2m'] - 273.15
df_hourly['wind_10m'] = np.sqrt(df_hourly['u10']**2 + df_hourly['v10']**2)
df_hourly['wind_2m'] = df_hourly['wind_10m'] * (4.87 / np.log(67.8 * 10 - 5.42))
df_hourly['ssrd_MJ'] = df_hourly['ssrd'] / 1000000.0  # J to MJ

# Resample to Daily
df_daily = pd.DataFrame()
df_daily['tmean'] = df_hourly['t2m_C'].resample('D').mean()
df_daily['tmax'] = df_hourly['t2m_C'].resample('D').max()
df_daily['tmin'] = df_hourly['t2m_C'].resample('D').min()
df_daily['tdew'] = df_hourly['d2m_C'].resample('D').mean()
df_daily['wind'] = df_hourly['wind_2m'].resample('D').mean()
df_daily['rs'] = df_hourly['ssrd_MJ'].resample('D').sum()

ds.close()

print("Calculating daily ET0...")
df_daily['day_of_year'] = df_daily.index.dayofyear
df_daily['ET0'] = calc_fao56_et0(
    df_daily['tmean'], df_daily['tmax'], df_daily['tmin'], df_daily['tdew'], 
    df_daily['wind'], df_daily['rs'], elevation_est, lat_ref, df_daily['day_of_year']
)

# ==========================================
# 4. WATER YEAR AGGREGATION & TREND ANALYSIS
# ==========================================
print("Aggregating to Water Years (June-May)...")
# Define Water Year: If month is < 6 (Jan-May), it belongs to the previous year
df_daily['WaterYear'] = np.where(df_daily.index.month < 6, df_daily.index.year - 1, df_daily.index.year)

# Keep only complete water years (365 or 366 days)
valid_years = df_daily.groupby('WaterYear').filter(lambda x: len(x) >= 365)

# Monthly and Annual Aggregations
monthly_et0 = valid_years.groupby(['WaterYear', valid_years.index.month])['ET0'].sum()
annual_et0 = valid_years.groupby('WaterYear')['ET0'].sum()

# Trend Analysis (Linear Regression)
x = annual_et0.index.values
y = annual_et0.values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("\n==========================================")
print("ANNUAL ET0 TREND ANALYSIS")
print("==========================================")
print(f"Study Period: {x[0]} to {x[-1]} Water Years")
print(f"Trend Slope: {slope:.4f} mm/year")
print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: The linear trend is STATISTICALLY SIGNIFICANT at the 5% level.")
else:
    print("Result: The linear trend is NOT statistically significant at the 5% level.")
print("==========================================\n")

# ==========================================
# 5. EXPORT & PLOT
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(x, y, marker='o', color='darkorange', linewidth=2, label='Annual ET0')
plt.plot(x, intercept + slope * x, 'r--', label=f'Trend: {slope:.2f} mm/yr (p={p_value:.3f})')
plt.title('Annual FAO-56 Reference Evapotranspiration (ET0) per Water Year')
plt.xlabel('Water Year (June - May)')
plt.ylabel('Annual ET0 (mm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plot_path = os.path.join(output_folder, "Task2_Annual_ET0_Plot.png")
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to: {plot_path}")
plt.show()

excel_path = os.path.join(output_folder, "Task2_ET0_Results.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    annual_df = pd.DataFrame({'Water_Year': x, 'Annual_ET0_mm': y})
    annual_df.to_excel(writer, sheet_name='Annual_ET0', index=False)
    
    monthly_df = monthly_et0.reset_index()
    monthly_df.columns = ['Water_Year', 'Calendar_Month', 'Monthly_ET0_mm']
    monthly_df.to_excel(writer, sheet_name='Monthly_ET0', index=False)
    
print(f"Task 2 calculations complete. Data exported to: {excel_path}")