# Step 1: Data Processing
import pandas as pd
data = pd.read_csv("Project 1 Data.csv")

# Step 2: Data Visualization
import numpy as np
import matplotlib.pyplot as plt
# Some quick look at the data
print(data.head())
print(data.columns)
# Histogram of Samples per Step
plt.title('Samples per Step')
plt.xlabel('Step')
plt.ylabel('Samples')
data['Step'].hist(bins=13, linewidth=0.5, edgecolor="white")
# Grouping and Info
grouped_data = data.groupby('Step').agg(['min', 'max'])
print("\nGrouped DataFrame with min & max values:")
print(grouped_data)
# Single plot with X, Y, Z overlaid
plt.figure(figsize=(10, 6))
plt.scatter(data['Step'], data['X'], label='X', color='blue')
plt.scatter(data['Step'], data['Y'], label='Y', color='orange')
plt.scatter(data['Step'], data['Z'], label='Z', color='green')
plt.title('Coordinate Values (X, Y, Z) vs Step')
plt.xlabel('Step')
plt.ylabel('Coordinate Value')
plt.legend()
plt.grid(True)
plt.show()