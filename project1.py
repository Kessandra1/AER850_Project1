# Step 1: Data Processing
import pandas as pd

# Read data
data = pd.read_csv("Project 1 Data.csv")




# Step 2: Data Visualization
print("\n-----Step 2: Data Visualization-----")
import numpy as np
import matplotlib.pyplot as plt

# Some quick look at the data
print("\nFirst 5 rows of data:")
print(data.head())
print(data.columns)

# Total number of samples (using rows in X column to count)
total_samples = np.size(data['X'].values)
print("\nTotal number of samples:", total_samples)

# Histogram of Samples per class (Step)
plt.title('Samples per Step')
plt.xlabel('Step')
plt.ylabel('Samples')
data['Step'].hist(bins=13, edgecolor="black")

# 3D scatter plot of Coordinates for each Step
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(data['X'], data['Y'], data['Z'], c=data['Step'], 
                     cmap='tab20')
ax.set_title('3D Scatter Plot of Coordinates by Step')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(scatter, label='Step')
plt.show()




# Step 3: Correlation Analysis
import seaborn as sns

# Looking at the colinearity of variables
corr_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 6))
plt.title("Correlation Matrix")
sns.heatmap(np.abs(corr_matrix))
