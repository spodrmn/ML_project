import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

# printing original table

print("This is the original dataset:\n")

# This shows all rows and columns. Hopefully it's not too big. Can edit otherwise.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)
