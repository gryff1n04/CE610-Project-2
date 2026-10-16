import os

# 1. Paste your exact API key between the quotes below
your_api_key = "f23e5182-5342-4e2e-90f8-0350795d8819"

# 2. This automatically creates the hidden .cdsapirc file in your Windows user folder
file_content = f"url: https://cds.climate.copernicus.eu/api\nkey: {your_api_key}"
file_path = r"C:\Users\yashs\.cdsapirc"

with open(file_path, "w") as f:
    f.write(file_content)

print(f"Success! Your Copernicus API key is saved to {file_path}")