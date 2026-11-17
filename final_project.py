import pandas as pd
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
print("Reading dataset and performing initial data cleaning...")

# Read the dataset
df = pd.read_csv("final_dataset.csv")
print("Original shape:", df.shape)
print("Class distribution ('round'):\n", df["round"].value_counts())
print("Duplicate rows found:", df.duplicated().sum())
print("Missing values found:", df.isna().sum().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after dropping duplicates:", df.shape[0])

# D1 - Original Dataset
print("\nPreparing D1 (Original Dataset)...")
D1 = df.copy()

# Separate features and target
df_features = df.iloc[:, :-1]
df_target = df.iloc[:, -1]

# D2 - Min-Max Scaled Dataset
print("Preparing D2 (Min-Max Scaled Dataset)...")
minmax_scaler = MinMaxScaler()
D2_features = minmax_scaler.fit_transform(df_features)
D2 = pd.DataFrame(D2_features, columns=df_features.columns)
D2["round"] = df_target.values

# D3 - Standard Scaled Dataset
print("Preparing D3 (Standard Scaled Dataset)...")
standard_scaler = StandardScaler()
D3_features = standard_scaler.fit_transform(df_features)
D3 = pd.DataFrame(D3_features, columns=df_features.columns)
D3["round"] = df_target.values

# Display the first 5 rows of each dataset
print("\nD1 - Original Dataset (First 5 rows):")
print(D1.head())

print("\nD2 - Min-Max Scaled Dataset (First 5 rows):")
print(D2.head())

print("\nD3 - Standard Scaled Dataset (First 5 rows):")
print(D3.head())


print("\n-----------------------------------PART 2: CLASSIFICATION-------------------------------------------")
print("Training classification models: K-Nearest Neighbors (KNN) and Random Forest...")

print("\n--- K Nearest Neighbors (KNN) ---")
print("Performing GridSearch to find the best neighbors (k) for each dataset...")

# D1 (Original)
X_d1 = D1.iloc[:, :-1]
y_d1 = D1.iloc[:, -1]

X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
    X_d1, y_d1, test_size=0.2, random_state=42, stratify=y_d1)

knn1 = GridSearchCV(KNeighborsClassifier(), {
                    'n_neighbors': [1, 3, 5, 7, 9, 11]}, cv=5)
knn1.fit(X_train_d1, y_train_d1)
pred_d1 = knn1.predict(X_test_d1)
print("D1 optimal classification k found:", knn1.best_params_['n_neighbors'])


# D2 (Min-Max Scaled)
minmax_clf_knn = MinMaxScaler()
X_train_d2 = pd.DataFrame(minmax_clf_knn.fit_transform(
    X_train_d1), columns=X_train_d1.columns)
X_test_d2 = pd.DataFrame(minmax_clf_knn.transform(
    X_test_d1), columns=X_test_d1.columns)
y_train_d2 = y_train_d1
y_test_d2 = y_test_d1

knn2 = GridSearchCV(KNeighborsClassifier(), {
                    'n_neighbors': [1, 3, 5, 7, 9, 11]}, cv=5)
knn2.fit(X_train_d2, y_train_d2)
pred_d2 = knn2.predict(X_test_d2)
print("D2 optimal classification k found:", knn2.best_params_['n_neighbors'])


# D3 (Standard Scaled)
std_clf_knn = StandardScaler()
X_train_d3 = pd.DataFrame(std_clf_knn.fit_transform(
    X_train_d1), columns=X_train_d1.columns)
X_test_d3 = pd.DataFrame(std_clf_knn.transform(
    X_test_d1), columns=X_test_d1.columns)
y_train_d3 = y_train_d1
y_test_d3 = y_test_d1

knn3 = GridSearchCV(KNeighborsClassifier(), {
                    'n_neighbors': [1, 3, 5, 7, 9, 11]}, cv=5)
knn3.fit(X_train_d3, y_train_d3)
pred_d3 = knn3.predict(X_test_d3)
print("D3 optimal classification k found:", knn3.best_params_['n_neighbors'])


print("\nGenerating confusion matrices for KNN...")
cm_d1 = confusion_matrix(y_test_d1, pred_d1)
cm_d2 = confusion_matrix(y_test_d2, pred_d2)
cm_d3 = confusion_matrix(y_test_d3, pred_d3)

acc_d1 = accuracy_score(y_test_d1, pred_d1)
acc_d2 = accuracy_score(y_test_d2, pred_d2)
acc_d3 = accuracy_score(y_test_d3, pred_d3)

print("KNN Accuracy Results:")
print(f"  D1 (Original): {acc_d1:.4f}")
print(f"  D2 (Min-Max):  {acc_d2:.4f}")
print(f"  D3 (Standard): {acc_d3:.4f}")

# Display KNN confusion matrices
labels = knn1.classes_
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

plt.tight_layout()
plt.show()


print("\n--- Random Forest Classification ---")
print("Training Random Forest models on all three datasets...")

# D1 (Original) - Random Forest
X_d1 = D1.iloc[:, :-1]
y_d1 = D1.iloc[:, -1]

X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
    X_d1, y_d1, test_size=0.2, random_state=42, stratify=y_d1)

print("Baseline accuracy (always guess the biggest class):",
      round(y_test_d1.value_counts(normalize=True).max(), 4))

rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train_d1, y_train_d1)
pred_rf_d1 = rf1.predict(X_test_d1)

# D2 (Min-Max Scaled) - Random Forest
minmax_clf_rf = MinMaxScaler()
X_train_d2 = pd.DataFrame(minmax_clf_rf.fit_transform(
    X_train_d1), columns=X_train_d1.columns)
X_test_d2 = pd.DataFrame(minmax_clf_rf.transform(
    X_test_d1), columns=X_test_d1.columns)
y_train_d2 = y_train_d1
y_test_d2 = y_test_d1

rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train_d2, y_train_d2)
pred_rf_d2 = rf2.predict(X_test_d2)

# D3 (Standard Scaled) - Random Forest
std_clf_rf = StandardScaler()
X_train_d3 = pd.DataFrame(std_clf_rf.fit_transform(
    X_train_d1), columns=X_train_d1.columns)
X_test_d3 = pd.DataFrame(std_clf_rf.transform(
    X_test_d1), columns=X_test_d1.columns)
y_train_d3 = y_train_d1
y_test_d3 = y_test_d1

rf3 = RandomForestClassifier(random_state=42)
rf3.fit(X_train_d3, y_train_d3)
pred_rf_d3 = rf3.predict(X_test_d3)

print("\nGenerating confusion matrices for Random Forest...")
cm_rf_d1 = confusion_matrix(y_test_d1, pred_rf_d1)
cm_rf_d2 = confusion_matrix(y_test_d2, pred_rf_d2)
cm_rf_d3 = confusion_matrix(y_test_d3, pred_rf_d3)

acc_rf_d1 = accuracy_score(y_test_d1, pred_rf_d1)
acc_rf_d2 = accuracy_score(y_test_d2, pred_rf_d2)
acc_rf_d3 = accuracy_score(y_test_d3, pred_rf_d3)

print("Random Forest Accuracy Results:")
print(f"  D1 (Original): {acc_rf_d1:.4f}")
print(f"  D2 (Min-Max):  {acc_rf_d2:.4f}")
print(f"  D3 (Standard): {acc_rf_d3:.4f}")

results = pd.DataFrame({
    "Model": ["KNN D1", "KNN D2", "KNN D3", "RF D1", "RF D2", "RF D3"],
    "Accuracy": [acc_d1, acc_d2, acc_d3, acc_rf_d1, acc_rf_d2, acc_rf_d3]})
print("\nAll models ranked by accuracy:")
print(results.sort_values("Accuracy", ascending=False).to_string(index=False))
best = results.sort_values("Accuracy", ascending=False).iloc[0]
print("\n=> Recommended model:", best["Model"],
      "| Accuracy:", round(best["Accuracy"], 3))

# Display RF confusion matrices
labels = rf1.classes_
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

plt.tight_layout()
plt.show()


print("\n-----------------------------------PART 3: CLUSTERING-------------------------------------------")
print("Testing K-Means clustering across k values from 2 to 9 to find optimal cluster counts...")

K_range = range(2, 10)

# CLUSTERING FOR D1 (Original Dataset)
print("\nEvaluating K-Means for D1 (Original Dataset)...")
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

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias_d1, marker='o')
plt.title('D1 Inertia')
plt.xlabel('k')
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes_d1, marker='o')
plt.title('D1 Silhouette')
plt.xlabel('k')
plt.tight_layout()
plt.show()

# Final D1 Model & Plot
print(f"Applying final K-Means model for D1 using optimal k={best_k_d1}...")
final_kmeans_d1 = KMeans(n_clusters=best_k_d1, random_state=42, n_init='auto')
final_kmeans_d1.fit(X_d1)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=D1, x='win_shares', y='average_minutes_played',
                hue=final_kmeans_d1.labels_, palette='viridis')
plt.title(
    f"D1 Final Clusters (k={best_k_d1} | Silhouette Score: {best_sil_score_d1:.4f})")
plt.show()

print("D1 clusters vs ground truth labels ('round'):")
print(pd.crosstab(final_kmeans_d1.labels_, D1['round']))

# CLUSTERING FOR D2 (Min-Max Scaled)
print("\nEvaluating K-Means for D2 (Min-Max Scaled Dataset)...")
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

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias_d2, marker='o')
plt.title('D2 Inertia')
plt.xlabel('k')
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes_d2, marker='o')
plt.title('D2 Silhouette')
plt.xlabel('k')
plt.tight_layout()
plt.show()

# Final D2 Model & Plot
print(f"Applying final K-Means model for D2 using optimal k={best_k_d2}...")
final_kmeans_d2 = KMeans(n_clusters=best_k_d2, random_state=42, n_init='auto')
final_kmeans_d2.fit(X_d2)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=D2, x='win_shares', y='average_minutes_played',
                hue=final_kmeans_d2.labels_, palette='viridis')
plt.title(
    f"D2 Final Clusters (k={best_k_d2} | Silhouette Score: {best_sil_score_d2:.4f})")
plt.show()

print("D2 clusters vs ground truth labels ('round'):")
print(pd.crosstab(final_kmeans_d2.labels_, D2['round']))

# CLUSTERING FOR D3 (Standard Scaled)
print("\nEvaluating K-Means for D3 (Standard Scaled Dataset)...")
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

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias_d3, marker='o')
plt.title('D3 Inertia')
plt.xlabel('k')
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes_d3, marker='o')
plt.title('D3 Silhouette')
plt.xlabel('k')
plt.tight_layout()
plt.show()

# Final D3 Model & Plot
print(f"Applying final K-Means model for D3 using optimal k={best_k_d3}...")
final_kmeans_d3 = KMeans(n_clusters=best_k_d3, random_state=42, n_init='auto')
final_kmeans_d3.fit(X_d3)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=D3, x='win_shares', y='average_minutes_played',
                hue=final_kmeans_d3.labels_, palette='viridis')
plt.title(
    f"D3 Final Clusters (k={best_k_d3} | Silhouette Score: {best_sil_score_d3:.4f})")
plt.show()

print("D3 clusters vs ground truth labels ('round'):")
print(pd.crosstab(final_kmeans_d3.labels_, D3['round']))


# --- NEWLY ADDED SUMMARY AND EXPLANATION BLOCK ---
print("\n-----------------------------------FINAL CLUSTERING SUMMARY-------------------------------------------")
print(
    f"Optimal 'k' for D1 (Original Dataset): {best_k_d1} (Silhouette Score: {best_sil_score_d1:.4f})")
print(
    f"Optimal 'k' for D2 (Min-Max Scaled):   {best_k_d2} (Silhouette Score: {best_sil_score_d2:.4f})")
print(
    f"Optimal 'k' for D3 (Standard Scaled):  {best_k_d3} (Silhouette Score: {best_sil_score_d3:.4f})")
print("\n------------------------------------------------------------------------------------------------------")


print("\n------------------------- PART 4: INSIGHTS, CONCLUSIONS & RECOMMENDATIONS -------------------------")
print("""
[ RESULTS ]
- Best Classification: KNN with Min-Max scaling (D2) achieved the highest accuracy at 61.34%, 
  providing a ~6% lift over the baseline majority guess (55.46%). Random Forest peaked on unscaled data (58.82%).
- Best Clustering: K-Means consistently identified K=2 as the optimal cluster count across all datasets. 
  While D1 had a massive silhouette score (0.7070), D2 and D3 stabilized around 0.46.

[ ANALYSIS & INSIGHTS ]
- Min-Max scaling significantly improved KNN because it relies on Euclidean distance; 
  without it, high-volume stats overpower efficiency stats. On the other hand, Random Forest is tree-based 
  and scale-invariant, which is why normalization did not improve its performance.
- D1's high silhouette score is caused by unscaled 
  volume metrics pulling players far apart. D2's score of ~0.46 represents a much more mathematically 
  sound grouping of players.
- The max accuracy of 61.3% proves that final career stats do not cleanly 
  separate into 1st/2nd round buckets. This highlights the unpredictability of the NBA draft, where 
  second-round players who overperform heavily blur the model's decision boundaries. 
  - e.g., Nikola Jokic (41st pick): 3x MVP, Finals MVP, 8x All-Star
  - e.g., Jalen Brunson (33rd pick): All-NBA, All-Star
  - e.g., Manu Ginobili (57th pick): Hall of Famer, 4x Champion

[ RECOMMENDATIONS FOR FUTURE WORK ]
1. Transition from raw totals to "rate" statistics (Points Per Minute, 
   Assist-to-Turnover Ratio, True Shooting %) which account for varying playing time and hold stronger signals.
2. Run a broader GridSearch on Random Forest (tuning n_estimators, max_depth, 
   min_samples_split) and test different distance metrics (Manhattan vs. Euclidean) for KNN.
3. Try other algorithms: Algorithms like XGBoost, LightGBM, or CatBoost build trees sequentially 
   (learning from previous errors) and consistently outperform Random Forest on tabular data.
---------------------------------------------------------------------------------------------------
""")
