import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(" -----------------------------------PART 1: NORMALIZATION-------------------------------------------\n")


D1 = pd.read_csv("final_dataset.csv")

# printing original table

print("This is the original dataset(D1):\n")

# This shows all rows and columns.
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)

print(D1)

# We have to separate the non-integer values, which are all located in the last column in this data set.

df_features = D1.iloc[:, :-1]
df_labels = D1.iloc[:, -1]

# Now, we will perform min-max scaling.

features_mmscaled = (df_features - df_features.min()) / \
    (df_features.max() - df_features.min())

features_mmscaled['round'] = df_labels.values

D2 = features_mmscaled

print("\nThese are the dataset's features, normalized by Min-Max Scaling(D2):'\n")
print(D2)

# Standard scaling is x minus mean divided by standard deviation.
# Pandas lets you use the mean and standard deviation of each column.
# IMPORTANT! We used population standard deviation (that's what the ddof=0 means), otherwise you might get different results.

features_stscaled = (df_features - df_features.mean()) / \
    df_features.std(ddof=0)

features_stscaled['round'] = df_labels.values

D3 = features_stscaled

print("\nThese are the dataset's features, normalized by Standard Scaling(D3):'\n")
print(features_stscaled)


print("\n-----------------------------------PART 2: CLASSIFICATION-------------------------------------------")

# We will now attempt two classification models for each dataset (D1, D2, D3).
# Specifically, K-nearest neighbors (KNN) and Random Forest Classification will be used.
# They will be implemented with sklearn

# Running K Nearest Neighbors:
# First, we'll find the optimal value for K for each test type using GridSearchCV

print("\n K Nearest Neighbors")
print("\n Looking for the optimal K value for each datasetusing GridSearchCV:")

datasets = {
    'D1 (Original)': D1,
    'D2 (Min-Max Scaled)': D2,
    'D3 (Standard Scaled)': D3
}

for name, data in datasets.items():
    print(f"-------- {name} --------")

    # Separate features (X) and label (y) for this specific dataset
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Setup GridSearchCV to find the best K between 1 and 30
    param_grid = {'n_neighbors': np.arange(1, 31)}
    knn_grid = GridSearchCV(KNeighborsClassifier(),
                            param_grid, cv=5, scoring='accuracy')

    # Train the models and find the optimal K
    knn_grid.fit(X_train, y_train)
    best_k = knn_grid.best_params_['n_neighbors']

    # Extract the absolute best model and make predictions on the test set
    best_model = knn_grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    print(f"Optimal K: {best_k}")
    print("\n")

print("\nNow we'll run KNN with the values we calculated for each of the datasets\n")
