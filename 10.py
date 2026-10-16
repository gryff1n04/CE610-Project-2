import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. INPUT DATA (Transcribed from your DDF Table)
# ==========================================
durations = [1, 2, 3, 5, 7, 10, 15, 30] # in days
return_periods = [2, 5, 10, 25, 50, 100] # in years

# Depths in mm
depths_data = {
    2:   [115.28, 160.10, 182.41, 211.84, 244.05, 286.06, 340.78, 461.70],
    5:   [167.05, 231.38, 261.15, 301.04, 340.41, 402.54, 465.90, 630.24],
    10:  [201.32, 278.57, 313.29, 360.10, 404.21, 479.66, 548.73, 741.83],
    25:  [244.63, 338.20, 379.15, 434.73, 484.82, 577.09, 653.40, 882.82],
    50:  [276.76, 382.43, 428.02, 490.09, 544.62, 649.38, 731.04, 987.41],
    100: [308.65, 426.34, 476.52, 545.04, 603.99, 721.13, 808.11, 1091.23]
}

df_depth = pd.DataFrame(depths_data, index=durations)

# ==========================================
# 2. CALCULATE INTENSITY
# ==========================================
# Intensity (mm/h) = Depth (mm) / (Duration in days * 24 hours)
df_intensity = df_depth.div(np.array(durations) * 24, axis=0)

# ==========================================
# 3. PLOTTING
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
markers = ['o', 's', '^', 'D', 'v', 'p']

# Plot 1: Depth-Duration-Frequency (DDF)
for i, T in enumerate(return_periods):
    ax1.plot(df_depth.index, df_depth[T], marker=markers[i], linewidth=2, label=f'T = {T} yrs')

ax1.set_title('Depth-Duration-Frequency (DDF) Curves', fontsize=14, fontweight='bold')
ax1.set_xlabel('Duration (Days)', fontsize=12)
ax1.set_ylabel('Rainfall Depth (mm)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(title='Return Period')

# Plot 2: Intensity-Duration-Frequency (IDF)
for i, T in enumerate(return_periods):
    ax2.plot(df_intensity.index, df_intensity[T], marker=markers[i], linewidth=2, label=f'T = {T} yrs')

ax2.set_title('Intensity-Duration-Frequency (IDF) Curves', fontsize=14, fontweight='bold')
ax2.set_xlabel('Duration (Days)', fontsize=12)
ax2.set_ylabel('Rainfall Intensity (mm/h)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(title='Return Period')

plt.tight_layout()
plt.savefig("Task1_DDF_IDF_Curves.png", dpi=300)
plt.show()