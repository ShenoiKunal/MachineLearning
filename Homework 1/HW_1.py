import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32



# Load data into PD dataframe
housing = pd.read_csv(Path("datasets/housing.csv"))

# Shuffle dataset, take the first x% determined by ratio, and split into test and train sets.
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Check if 32-bit checksum is in the within the test ratio set (ratio * 2^32)
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

# Split data based on hashed ID
def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Cut set into bins based on median_income
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])

# Create bar graph
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income Category")
plt.ylabel("Number of districts")
plt.show()