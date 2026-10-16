import imdlib as imd
import os

target_folder = r"D:\IITK\Sem 1\CE610\P2\Rdata"
os.makedirs(target_folder, exist_ok=True)

start_year = 1951
end_year = 2025

print(f"Starting reliable download via imdlib to {target_folder}...")

for year in range(start_year, end_year + 1):
    try:
        # Load data for a single year
        data = imd.load('rain', year, year)
        
        # Convert the binary data to an xarray dataset
        ds = data.get_xarray()
        
        # Define the output file path to exactly match the manual download format
        file_name = f"RF25_ind{year}_rfp25.nc"
        file_path = os.path.join(target_folder, file_name)
        
        # Save as a NetCDF file
        ds.to_netcdf(file_path)
        
        print(f"Successfully saved {file_name}")
    
    except Exception as e:
        print(f"Error processing {year}: {e}")

print("All years processed.")