# Step 1: Data Processing
import pandas as pd
data = pd.read_csv("Project 1 Data.csv")

# Step 2: Data Visualization
print("\n-----Step 2: Data Visualization-----")
import numpy as np
import matplotlib.pyplot as plt
# Some quick look at the data
print("\nFirst 5 rows of data:")
print(data.head())
print(data.columns)
# Histogram of Samples per Step
plt.title('Samples per Step')
plt.xlabel('Step')
plt.ylabel('Samples')
data['Step'].hist(bins=13, linewidth=0.5, edgecolor="white")
# Total number of samples (using X column to count)
total_samples = np.size(data[['X']].values)
print("\nTotal number of coordinate values:", total_samples)
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

# Step 3: Correlation Analysis
print("\n-----Step 3: Correlation Analysis-----")
import seaborn as sns
# Looking at the colinearity of variables
corr_matrix = data.corr()
# Plot the correlation matrix
plt.title("Correlation Matrix")
sns.heatmap(np.abs(corr_matrix))
# Each feature's (X, Y, and Z) correlation with the target variable (Step)
step_corr = corr_matrix['Step'].drop('Step')
print("\nCorrelation of each feature with 'Step':")
print(step_corr)