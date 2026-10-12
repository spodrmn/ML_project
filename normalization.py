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

print("\nThese are the dataset's features, separated for normalization.'\n")

print(df_features)
