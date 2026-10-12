import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Classification Method #1 K Nearest Neighbors!

# This is a dataset with offensive/defensive ratings of various NBA players and whether or not they are All-Stars

df = pd.read_csv("classification_test.csv")

print("This is the original dataset:\n")
print(df)

# Split features and labels

df_features = df.iloc[:, :-1]
df_labels = df['Label']

print("\nThese are the features:")
print(df_features)

print("\nThese are the labels:")
print(df_labels)

# Running K Nearest Neighbors with k=3

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(df_features, df_labels)


print("\nK Nearest Neighbors Results:")

label_pred = knn.predict(df_features)

print(label_pred)

# Classification Method #2 Random Forest Regressor (Decision Trees) !

rf_reg = RandomForestClassifier(n_estimators=100, random_state=42)

rf_reg.fit(df_features, df_labels)

rf_pred = rf_reg.predict(df_features)

print("\nRandom Forest Results:")
print(rf_pred)
