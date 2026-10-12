import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# This is a dataset containing offensive/defensive ratings of various NBA players

df = pd.read_csv("clustering_test.csv")

print(df)

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(df)

plt.scatter(df['Offense'], df['Defense'])
plt.show()
