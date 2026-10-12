import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

# printing original table

print("This is the original dataset:\n")

# This shows all rows and columns. Hopefully it's not too big. Can edit otherwise.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)

# We have to separate the non-integer values.
# For the sake of my sanity, I will assume our dataset will have all labels on the last column.

df_features = df.iloc[:, :-1]

# I will print to check.

print("\nThese are the dataset's features, separated for normalization:'\n")

print(df_features)

# If we have to combine them we can just edit the original dataset later.
# Now, I will try min-max scaling.

features_mmscaled = (df_features - df_features.min()) / \
    (df_features.max() - df_features.min())

print("\nThese are the dataset's features, normalized by Min-Max Scaling:'\n")
print(features_mmscaled)

# As far as I can tell it seems correct. I will proceed.

# Standard scaling is x minus mean divided by standard deviation
