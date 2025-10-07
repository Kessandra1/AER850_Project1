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




# Step 4: Classification Model Development/Engineering
print("\n-----Step 4: Classification Model Development/Engineering-----")

# Prepare data using Test-Train split
from sklearn.model_selection import train_test_split
X = data.drop('Step', axis=1)
y = data['Step']
# Split data using Stratification for better distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Assess each model's best hyperparameters using grid search cross-validation
from sklearn.model_selection import GridSearchCV

# Model 1: Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Define pipeline with scaler and classifier
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
# Define hyperparameter grid
param1 = {
    "clf__C": [100, 10000000, 100000000]
}
# Utilize grid search cross-validation
gs1 = GridSearchCV(pipe, param_grid=param1, scoring="neg_log_loss", cv=5, 
                   n_jobs=-1)
gs1.fit(X_train, y_train)
clf1 = gs1.best_estimator_
print("\nLogistic Regression's best parameters (GridSearchCV):", 
      gs1.best_params_)

# Model 2: Decision Trees
from sklearn.tree import DecisionTreeClassifier
# Define classifier
clf2 = DecisionTreeClassifier(class_weight='balanced', random_state=42)
# Define hyperparameter grid
param2 = {
    "max_depth": [3, 4],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt"]
}
# Utilize grid search cross-validation
gs2 = GridSearchCV(clf2, param_grid=param2, scoring="neg_log_loss", cv=5, 
                   n_jobs=-1)
gs2.fit(X_train, y_train)
clf2 = gs2.best_estimator_
print("\nDecision Tree Classifier's best parameters (GridSearchCV):", 
      gs2.best_params_)

# Model 3: Random Forest
from sklearn.ensemble import RandomForestClassifier
# Define classifier
clf3 = RandomForestClassifier(class_weight='balanced', random_state=42)
from sklearn.ensemble import RandomForestClassifier
# Define hyperparameter grid
param3 = {
    "n_estimators": [10, 30],
    "max_depth": [None, 2],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini", "entropy"],
    "max_features": [None, "sqrt"]
}
# Utilize grid search cross-validation
gs3 = GridSearchCV(clf3, param_grid=param3, scoring="neg_log_loss", cv=5, 
                   n_jobs=-1)
gs3.fit(X_train, y_train)
clf3 = gs3.best_estimator_
print("\nRandom Forest Classifier's best parameters (GridSearchCV):", 
      gs3.best_params_)
# Utilize RandomizedSearchCV.
from sklearn.model_selection import RandomizedSearchCV
rs3 = RandomizedSearchCV(clf3, param_distributions=param3, n_iter=20, 
                         scoring="neg_log_loss", cv=5, n_jobs=-1, 
                         random_state=42)
rs3.fit(X_train, y_train)
clf3 = rs3.best_estimator_
print("Random Forest Classifier's best parameters (RandomizedSearchCV):", 
      rs3.best_params_)




# Step 5: Model Performance Analysis
print("\n-----Step 5: Model Performance Analysis-----")

# Determine each model's f1 score, precision, and accuracy
from sklearn.metrics import f1_score, precision_score, accuracy_score

# Model 1: Logistic Regression
y_pred_clf1 = clf1.predict(X_test)
f1_clf1 = f1_score(y_test, y_pred_clf1, average='weighted', zero_division=0)
precision_clf1 = precision_score(y_test, y_pred_clf1, average='weighted', 
                                 zero_division=0)
accuracy_clf1 = accuracy_score(y_test, y_pred_clf1)
# Print metrics to compare to other models
print("\nModel 1 - F1 Score:", f1_clf1)
print("        - Precision:", precision_clf1)
print("        - Accuracy:", accuracy_clf1)

# Model 2: Decision Trees
y_pred_clf2 = clf2.predict(X_test)
f1_clf2 = f1_score(y_test, y_pred_clf2, average='weighted', zero_division=0)
precision_clf2 = precision_score(y_test, y_pred_clf2, average='weighted', 
                                 zero_division=0)
accuracy_clf2 = accuracy_score(y_test, y_pred_clf2)
# Print metrics to compare to other models
print("Model 2 - F1 Score:", f1_clf2)
print("        - Precision:", precision_clf2)
print("        - Accuracy:", accuracy_clf2)

# Model 3: Random Forest
y_pred_clf3 = clf3.predict(X_test)
f1_clf3 = f1_score(y_test, y_pred_clf3, average='weighted', zero_division=0)
precision_clf3 = precision_score(y_test, y_pred_clf3, average='weighted', 
                                 zero_division=0)
accuracy_clf3 = accuracy_score(y_test, y_pred_clf3)
# Print metrics to compare to other models
print("Model 3 - F1 Score:", f1_clf3)
print("        - Precision:", precision_clf3)
print("        - Accuracy:", accuracy_clf3)

# Create confusion matrix (normalize for better visuals) based on model 3
from sklearn.metrics import confusion_matrix
cm_clf3 = confusion_matrix(y_test, y_pred_clf3, normalize='true')
# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_clf3, annot=True)
plt.title("Confusion Matrix for Model 3 (Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
