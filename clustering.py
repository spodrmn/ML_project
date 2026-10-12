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
plt.title("After K-Means (k=3)")


# Now I'll run K-Means for different values of k and plot their inertia

# I will store each k-inertia pair in a dictionary then turn that into a dataframe then plot it.

data_list = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init='auto')
    kmeans.fit(df)
    inert = kmeans.inertia_

    data_list.append({'k': i, 'inertia': inert})

elbow_graph = pd.DataFrame(data_list)

plt.figure(3)
sns.scatterplot(data=elbow_graph, x='k', y='inertia')
plt.title("Elbow Method Graph")

plt.show()
