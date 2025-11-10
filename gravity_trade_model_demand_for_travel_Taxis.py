# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:01:17 2025

@author: ARCA_5
"""


import pandas as pd
import numpy as np
import math

# === Load and clean the data ===
file_path = "E://Projects_to_be_applied//Gravity trade model//Actual_Data_9-11-2025//Data for Gravety Model - Copy.xlsx"

# Read the sheet without assuming a header
raw = pd.read_excel(file_path, sheet_name="Taxis", header=None)

# Find the header row (the one that contains "City" and "population")
header_row = raw[raw.apply(lambda r: r.astype(str).str.contains("City", case=False).any(), axis=1)].index[0]

# Re-read using that row as the header
df = pd.read_excel(file_path, sheet_name="Taxis", header=header_row)

# Drop empty columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Extract list of city columns (distance matrix headers)
cities = df.columns[2:]  # first two are "City" and "population"

# Clean data
df = df.dropna(subset=["City"]).reset_index(drop=True)

# Convert distances to numeric
distance_matrix = df[cities].apply(pd.to_numeric, errors="coerce").to_numpy()

# Extract populations
populations = df["population"].astype(float).to_numpy()

# === Gravity model computation ===
alpha, beta, gamma = 1, 1, 2  # model parameters
n = len(cities)
flow_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j and distance_matrix[i, j] > 0:
            flow_matrix[i, j] = (populations[i] ** alpha) * (populations[j] ** beta) / (distance_matrix[i, j] ** gamma)

# Avoid zeros
flow_matrix[flow_matrix <= 0] = 1e-6

# Normalize to total number of cars
total_cars = 7176
flow_matrix *= total_cars / flow_matrix.sum()

# === Round down (floor) ===
flow_matrix = np.ceil(flow_matrix)

# === Zero out diagonal and lower triangle ===
for i in range(n):
    for j in range(n):
        if i == j or i > j:  # diagonal and below diagonal
            flow_matrix[i, j] = 0
        elif flow_matrix[i, j] < 1:
            flow_matrix[i, j] = 5  # ensure positive flow above diagonal

# === Renormalize to keep total cars = 6548 ===
current_total = flow_matrix.sum()
if current_total != total_cars:
    flow_matrix *= total_cars / current_total

# Round again to keep integers after scaling
flow_matrix = np.round(flow_matrix).astype(int)

# === Adjust to match total exactly (fix rounding drift) ===
diff = int(flow_matrix.sum() - total_cars)
if diff != 0:
    flat = flow_matrix.flatten()
    idx = np.argsort(-flat)  # descending order
    for k in range(abs(diff)):
        if diff > 0:
            # Too many cars → subtract 1 from largest flow > 1
            if flat[idx[k % len(flat)]] > 1:
                flat[idx[k % len(flat)]] -= 1
        else:
            # Too few cars → add 1 to largest flow
            flat[idx[k % len(flat)]] += 1
    flow_matrix = flat.reshape(flow_matrix.shape)

# Double-check diagonal and lower triangle remain 0
for i in range(n):
    for j in range(n):
        if i == j or i > j:
            flow_matrix[i, j] = 0

# === Prepare DataFrame ===
flow_df = pd.DataFrame(flow_matrix, columns=cities, index=cities)

# Add total outflow per city and total inflow row
flow_df["Total Outflow"] = flow_df.sum(axis=1)
totals_row = flow_df.sum(axis=0)
flow_df.loc["Total Inflow"] = totals_row

# === Save results ===
output_file = "gravity_trade_model_actual_data_Ceil_results_Taxis.xlsx"
flow_df.to_excel(output_file)

print(f"✅ Gravity model estimation completed and saved as '{output_file}'")
print(f"Total cars in output: {flow_matrix.sum():,.0f}")
