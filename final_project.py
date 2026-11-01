import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


print("-----------------------------------PART 1: NORMALIZATION-------------------------------------------")

# Read the dataset
df = pd.read_csv("final_dataset.csv")

# D1 - Original Dataset
D1 = df.copy()

# Separate features and target
df_features = df.iloc[:, :-1]
df_target = df.iloc[:, -1]

# D2 - Min-Max Scaled Dataset
minmax_scaler = MinMaxScaler()

D2_features = minmax_scaler.fit_transform(df_features)

D2 = pd.DataFrame(D2_features, columns=df_features.columns)
D2["round"] = df_target.values

# D3 - Standard Scaled Dataset
standard_scaler = StandardScaler()

D3_features = standard_scaler.fit_transform(df_features)

D3 = pd.DataFrame(D3_features, columns=df_features.columns)
D3["round"] = df_target.values

# Display the first 5 rows of each dataset
print("\nD1 - Original Dataset")
print(D1.head())

print("\nD2 - Min-Max Scaled Dataset")
print(D2.head())

print("\nD3 - Standard Scaled Dataset")
print(D3.head())


print("\n-----------------------------------PART 2: CLASSIFICATION-------------------------------------------")

# We will now attempt two classification models for each dataset (D1, D2, D3).
# Specifically, K-nearest neighbors (KNN) and Random Forest Classification will be used.
# They will be implemented with sklearn

# Running K Nearest Neighbors:
# We will use k=3, since it performed the best during our testing (using elbow method to check)

print("\n K Nearest Neighbors (k=3)")

# D1 (Original)
X_d1 = D1.iloc[:, :-1]
y_d1 = D1.iloc[:, -1]

# Split into training and testing sets
# Added startify=y_d1 to ensure that the class distribution is preserved in both training and testing sets
X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
    X_d1, y_d1, test_size=0.2, random_state=42, stratify=y_d1)

knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train_d1, y_train_d1)        # Train only on the training set
pred_d1 = knn1.predict(X_test_d1)       # Predict only on the testing set


# D2 (Min-Max Scaled)
X_d2 = D2.iloc[:, :-1]
y_d2 = D2.iloc[:, -1]

X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
    X_d2, y_d2, test_size=0.2, random_state=42)

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train_d2, y_train_d2)
pred_d2 = knn2.predict(X_test_d2)


# D3 (Standard Scaled)
X_d3 = D3.iloc[:, :-1]
y_d3 = D3.iloc[:, -1]

X_train_d3, X_test_d3, y_train_d3, y_test_d3 = train_test_split(
    X_d3, y_d3, test_size=0.2, random_state=42)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train_d3, y_train_d3)
pred_d3 = knn3.predict(X_test_d3)


print("D1 Predictions (First 20):")
print(pred_d1[:20])

print("\nD2 Predictions (First 20):")
print(pred_d2[:20])

print("\nD3 Predictions (First 20):")
print(pred_d3[:20])

# Generating confusion matrices for each
cm_d1 = confusion_matrix(y_test_d1, pred_d1)
cm_d2 = confusion_matrix(y_test_d2, pred_d2)
cm_d3 = confusion_matrix(y_test_d3, pred_d3)

# Get the unique class labels (e.g., 'First', 'Second') to label the boxes
labels = knn1.classes_

# Set up a plotting area with 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm_d1, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('D1 (Original) - KNN')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_d2, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('D2 (Min-Max Scaled) - KNN')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

sns.heatmap(cm_d3, annot=True, fmt='d', cmap='Oranges',
            xticklabels=labels, yticklabels=labels, ax=axes[2])
axes[2].set_title('D3 (Standard Scaled) - KNN')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')

# Display the graphics
plt.tight_layout()
plt.show()


print("\n Random Forest Classification")

# D1 (Original) - Random Forest
X_d1 = D1.iloc[:, :-1]
y_d1 = D1.iloc[:, -1]

X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
    X_d1, y_d1, test_size=0.2, random_state=42)

rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train_d1, y_train_d1)
pred_rf_d1 = rf1.predict(X_test_d1)

# D2 (Min-Max Scaled) - Random Forest
X_d2 = D2.iloc[:, :-1]
y_d2 = D2.iloc[:, -1]

X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
    X_d2, y_d2, test_size=0.2, random_state=42)

rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train_d2, y_train_d2)
pred_rf_d2 = rf2.predict(X_test_d2)

# D3 (Standard Scaled) - Random Forest

X_d3 = D3.iloc[:, :-1]
y_d3 = D3.iloc[:, -1]

X_train_d3, X_test_d3, y_train_d3, y_test_d3 = train_test_split(
    X_d3, y_d3, test_size=0.2, random_state=42)

rf3 = RandomForestClassifier(random_state=42)
rf3.fit(X_train_d3, y_train_d3)
pred_rf_d3 = rf3.predict(X_test_d3)


print("D1 Random Forest Predictions (First 20):")
print(pred_rf_d1[:20])

print("\nD2 Random Forest Predictions (First 20):")
print(pred_rf_d2[:20])

print("\nD3 Random Forest Predictions (First 20):")
print(pred_rf_d3[:20])

# Generating confusion matrices for each
cm_rf_d1 = confusion_matrix(y_test_d1, pred_rf_d1)
cm_rf_d2 = confusion_matrix(y_test_d2, pred_rf_d2)
cm_rf_d3 = confusion_matrix(y_test_d3, pred_rf_d3)

# Get the unique class labels (e.g., 'First', 'Second') to label the boxes
labels = rf1.classes_

# Set up a plotting area with 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm_rf_d1, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('D1 (Original) - Random Forest')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_rf_d2, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('D2 (Min-Max Scaled) - Random Forest')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

sns.heatmap(cm_rf_d3, annot=True, fmt='d', cmap='Oranges',
            xticklabels=labels, yticklabels=labels, ax=axes[2])
axes[2].set_title('D3 (Standard Scaled) - Random Forest')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')

# Display the graphics
plt.tight_layout()
plt.show()


print("\n-----------------------------------PART 3: CLUSTERING-------------------------------------------")
K_range = range(2, 10)
# CLUSTERING FOR D1 (Original Dataset)
X_d1 = D1.iloc[:, :-1]  # Drop label column

inertias_d1 = []
silhouettes_d1 = []
best_k_d1 = 2
best_sil_score_d1 = -1

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_temp.fit(X_d1)

    inertias_d1.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(X_d1, kmeans_temp.labels_)
    silhouettes_d1.append(sil_score)

    if sil_score > best_sil_score_d1:
        best_sil_score_d1 = sil_score
        best_k_d1 = k

# Final D1 Model & Plot
final_kmeans_d1 = KMeans(n_clusters=best_k_d1, random_state=42, n_init='auto')
final_kmeans_d1.fit(X_d1)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=D1, x='win_shares', y='average_minutes_played',
                hue=final_kmeans_d1.labels_, palette='viridis')
plt.title(
    f"D1 Final Clusters (k={best_k_d1} | Silhouette Score: {best_sil_score_d1:.4f})")
plt.show()


# CLUSTERING FOR D2 (Min-Max Scaled)
X_d2 = D2.iloc[:, :-1]  # Drop label column

inertias_d2 = []
silhouettes_d2 = []
best_k_d2 = 2
best_sil_score_d2 = -1

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_temp.fit(X_d2)

    inertias_d2.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(X_d2, kmeans_temp.labels_)
    silhouettes_d2.append(sil_score)

    if sil_score > best_sil_score_d2:
        best_sil_score_d2 = sil_score
        best_k_d2 = k

# Final D2 Model & Plot
final_kmeans_d2 = KMeans(n_clusters=best_k_d2, random_state=42, n_init='auto')
final_kmeans_d2.fit(X_d2)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=D2, x='win_shares', y='average_minutes_played',
                hue=final_kmeans_d2.labels_, palette='viridis')
plt.title(
    f"D2 Final Clusters (k={best_k_d2} | Silhouette Score: {best_sil_score_d2:.4f})")
plt.show()


# CLUSTERING FOR D3 (Standard Scaled)
X_d3 = D3.iloc[:, :-1]  # Drop label column

inertias_d3 = []
silhouettes_d3 = []
best_k_d3 = 2
best_sil_score_d3 = -1

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_temp.fit(X_d3)

    inertias_d3.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(X_d3, kmeans_temp.labels_)
    silhouettes_d3.append(sil_score)

    if sil_score > best_sil_score_d3:
        best_sil_score_d3 = sil_score
        best_k_d3 = k

# Final D3 Model & Plot
final_kmeans_d3 = KMeans(n_clusters=best_k_d3, random_state=42, n_init='auto')
final_kmeans_d3.fit(X_d3)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=D3, x='win_shares', y='average_minutes_played',
                hue=final_kmeans_d3.labels_, palette='viridis')
plt.title(
    f"D3 Final Clusters (k={best_k_d3} | Silhouette Score: {best_sil_score_d3:.4f})")
plt.show()
