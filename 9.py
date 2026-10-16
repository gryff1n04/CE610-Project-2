import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. INPUT PARAMETERS
# ==========================================
# Update this path to your extracted 2024-2025 daily rainfall CSV
csv_file_path = "D:\IITK\Sem 1\CE610\P2\Rdata\water_year_24_25.csv" 
CN = 64.3

# ==========================================
# 2. SCS-CN CALCULATIONS
# ==========================================
# Maximum potential retention (S) in mm
S = (25400 / CN) - 254
# Initial abstraction (Ia)
Ia = 0.2 * S

# Load daily rainfall data
# Ensure CSV has 'Date' and 'Rainfall_mm' columns
df = pd.read_csv(csv_file_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Calculate Daily Runoff (Q)
# Q = (P - Ia)^2 / (P - Ia + S) for P > Ia, else Q = 0
df['Runoff_mm'] = np.where(
    df['Rainfall_mm'] > Ia,
    ((df['Rainfall_mm'] - Ia)**2) / (df['Rainfall_mm'] - Ia + S),
    0
)

# Calculate Monthly Runoff (Sum of daily values)
monthly_summary = df.resample('ME').sum()

# Annual Totals
annual_rainfall = df['Rainfall_mm'].sum()
annual_runoff = df['Runoff_mm'].sum()
runoff_coefficient = annual_runoff / annual_rainfall if annual_rainfall > 0 else 0

print("==========================================")
print("TASK 5: RUNOFF ESTIMATION RESULTS")
print("==========================================")
print(f"Composite CN: {CN}")
print(f"Total Annual Rainfall: {annual_rainfall:.2f} mm")
print(f"Total Annual Runoff: {annual_runoff:.2f} mm")
print(f"Annual Runoff Coefficient: {runoff_coefficient:.4f}")
print("==========================================")

# ==========================================
# 3. PLOTTING
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Daily Runoff
ax1.plot(df.index, df['Runoff_mm'], color='blue', linewidth=1.5)
ax1.set_title('Daily Direct Runoff (Water Year 2024-2025)')
ax1.set_ylabel('Runoff (mm)')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Monthly Runoff
# Format month names for the x-axis
month_labels = [month.strftime('%b %Y') for month in monthly_summary.index]
ax2.bar(month_labels, monthly_summary['Runoff_mm'], color='teal', edgecolor='black')
ax2.set_title('Monthly Direct Runoff (Water Year 2024-2025)')
ax2.set_ylabel('Runoff (mm)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("Task5_Runoff_Plots.png", dpi=300)
plt.show()