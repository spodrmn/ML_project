import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


# This is a dataset containing offensive/defensive ratings of various NBA players

df = pd.read_csv("clustering_test.csv")

print(df)

plt.figure(1)
sns.scatterplot(data=df, x='Offense', y='Defense')
plt.title("Original Dataset")


# Running K-Means where k=3

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(df)

# Clusters after K-Means
plt.figure(2)
sns.scatterplot(data=df, x='Offense', y='Defense', hue=kmeans.labels_)
plt.title("After K-Means")

plt.show()
