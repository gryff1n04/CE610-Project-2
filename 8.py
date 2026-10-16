import xarray as xr
import pandas as pd
import os

# ==========================================
# 1. FILE PATHS & COORDINATES
# ==========================================
# Update these paths to match your downloaded IMD NetCDF files
nc_2024 = r"D:\IITK\Sem 1\CE610\P2\Rdata\RF25_ind2024_rfp25.nc"
nc_2025 = r"D:\IITK\Sem 1\CE610\P2\Rdata\RF25_ind2025_rfp25.nc"

lat_ref = 23.75
lon_ref = 74.25

print("Loading and combining NetCDF files...")
ds_2024 = xr.open_dataset(nc_2024)
ds_2025 = xr.open_dataset(nc_2025)

# Concatenate along the uppercase 'TIME' dimension
ds_combined = xr.concat([ds_2024, ds_2025], dim='TIME')

# ==========================================
# 2. DATA EXTRACTION
# ==========================================
print(f"Extracting rainfall for coordinates (Lat: {lat_ref}, Lon: {lon_ref})...")
# Use the uppercase dimension names from the IMD dataset
ds_point = ds_combined.sel(LATITUDE=lat_ref, LONGITUDE=lon_ref, method='nearest')

# Convert to DataFrame
df = ds_point.to_dataframe().reset_index()

# Handle uppercase time column
time_col = 'TIME'
df[time_col] = pd.to_datetime(df[time_col])

# ==========================================
# 3. FILTER WATER YEAR & EXPORT
# ==========================================
print("Filtering for the 2024-2025 Water Year (June 1, 2024 - May 31, 2025)...")
mask = (df[time_col] >= '2024-06-01') & (df[time_col] <= '2025-05-31')
df_water_year = df.loc[mask].copy()

# Automatically identify the rainfall variable column by excluding coordinates
rain_col = [c for c in df_water_year.columns if c not in ['LATITUDE', 'LONGITUDE', 'TIME']][0]

# Format into the final CSV structure required by the SCS-CN script
df_final = pd.DataFrame({
    'Date': df_water_year[time_col].dt.date,
    'Rainfall_mm': df_water_year[rain_col]
})

output_csv = os.path.join(os.path.dirname(nc_2024), "water_year_24_25.csv")
df_final.to_csv(output_csv, index=False)

print(f"Extraction complete! Saved {len(df_final)} days of data to:")
print(output_csv)