import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==========================================
# 1. INPUT PARAMETERS (Loam Soil & Storm)
# ==========================================
K = 3.4          # Hydraulic conductivity (mm/h)
psi = 88.9       # Wetting front suction head (mm)
theta_s = 0.463  # Saturated moisture content
theta_i = 0.2    # Initial moisture content (from instructions)
d_theta = theta_s - theta_i

P_total = 276.76 # 50-yr 24-h rainfall depth (mm)
t_storm = 24.0   # Storm duration (hours)
i = P_total / t_storm # Rainfall intensity (mm/h)

# ==========================================
# 2. GREEN-AMPT CALCULATIONS
# ==========================================
# Calculate Ponding Time (tp) and Infiltration at Ponding (Fp)
tp = (K * psi * d_theta) / (i * (i - K))
Fp = i * tp

# Time array for the storm duration
times = np.linspace(0.01, t_storm, 500)
F_t = np.zeros_like(times)
f_t = np.zeros_like(times)

# Function to solve implicit Green-Ampt equation for post-ponding
def green_ampt_implicit(F, t, tp, Fp, K, psi, d_theta):
    return F - Fp - K * (t - tp) - psi * d_theta * np.log((psi * d_theta + F) / (psi * d_theta + Fp))

for idx, t in enumerate(times):
    if t < tp:
        # Pre-ponding
        F_t[idx] = i * t
        f_t[idx] = i
    else:
        # Post-ponding (solve implicitly)
        F_guess = Fp + K * (t - tp) # initial guess
        F_sol = fsolve(green_ampt_implicit, F_guess, args=(t, tp, Fp, K, psi, d_theta))[0]
        F_t[idx] = F_sol
        f_t[idx] = K * (1 + (psi * d_theta) / F_sol)

# Total infiltration loss is the cumulative infiltration at the end of the storm
total_infiltration = F_t[-1]
infiltration_pct = (total_infiltration / P_total) * 100

print("==========================================")
print("TASK 4: GREEN-AMPT INFILTRATION RESULTS")
print("==========================================")
print(f"Rainfall Intensity: {i:.2f} mm/h")
print(f"Ponding Time: {tp:.3f} hours ({tp*60:.1f} minutes)")
print(f"Total Infiltration Loss: {total_infiltration:.2f} mm")
print(f"Infiltration as % of Event Rainfall: {infiltration_pct:.1f}%")
print("==========================================")

# ==========================================
# 3. PLOTTING
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(times, f_t, 'b-', linewidth=2, label='Infiltration Rate (f)')
ax1.axhline(K, color='b', linestyle='--', alpha=0.5, label='Hydraulic Cond. (K)')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Infiltration Rate (mm/h)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim(0, max(i * 1.1, f_t.max() * 1.1))

ax2 = ax1.twinx()
ax2.plot(times, F_t, 'g-', linewidth=2, label='Cumulative Infiltration (F)')
ax2.set_ylabel('Cumulative Infiltration (mm)', color='g')
ax2.tick_params('y', colors='g')
ax2.set_ylim(0, P_total)

plt.title('Green-Ampt Infiltration (24-h, 50-yr Storm on Loam Soil)')
fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig("Task4_Infiltration_Curves.png", dpi=300)
plt.show()