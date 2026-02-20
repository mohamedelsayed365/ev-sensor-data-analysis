""""
Project: Simulated Electric Vehicle Sensor Data Analysis
Author: Mohamed Elsayed
Description:
This project simulates electric vehicle sensor data and performs
exploratory data analysis using NumPy, Pandas, and Matplotlib.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. SIMULATE SENSOR DATA
# ============================================================

np.random.seed(42)

num_samples = 1000

speed = np.abs(np.random.normal(80, 20, num_samples))        # km/h
battery_temp = np.random.normal(35, 5, num_samples)          # °C
torque = np.random.normal(250, 50, num_samples)              # Nm
soc = np.linspace(100, 20, num_samples) + np.random.normal(0, 2, num_samples)
acceleration = np.random.normal(0, 2, num_samples)

data = pd.DataFrame({
    "Speed_kmh": speed,
    "Battery_Temp_C": battery_temp,
    "Torque_Nm": torque,
    "State_of_Charge_%": soc,
    "Acceleration_m_s2": acceleration
})

print(data.head())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\nChecking for missing values:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())


# ============================================================
# 4. BASIC STATISTICAL ANALYSIS
# ============================================================

mean_speed = data["Speed_kmh"].mean()
max_temp = data["Battery_Temp_C"].max()

print(f"\nAverage Speed: {mean_speed:.2f} km/h")
print(f"Maximum Battery Temperature: {max_temp:.2f} °C")


# ============================================================
# 5. DATA VISUALIZATION
# ============================================================

plt.figure()
plt.hist(data["Speed_kmh"], bins=30)
plt.title("Speed Distribution")
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.scatter(data["Speed_kmh"], data["Battery_Temp_C"])
plt.title("Speed vs Battery Temperature")
plt.xlabel("Speed (km/h)")
plt.ylabel("Battery Temperature (°C)")
plt.show()


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
