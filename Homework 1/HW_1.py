import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Load data into PD dataframe
housing = pd.read_csv(Path("datasets/housing.csv"))


# Check if 32-bit checksum is in the within the test ratio set (ratio * 2^32)
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

# Show that the random samples are always the same with the same random_state
#     trainSplit, testSplit = train_test_split(housing, test_size=0.2, random_state=42)
#     print("First Random Training Sample (Rows 1-5)")
#     print(trainSplit[1:5])
#     print("First Random Test Sample (Rows 1-5)")
#     print(testSplit[1:5])
#     second_trainSplit, second_testSplit = train_test_split(housing, test_size=0.2, random_state=42)
#     print("Second Random Training Sample (Rows 1-5)")
#     print(second_trainSplit[1:5])
#     print("Second Random Test Sample (Rows 1-5)")
#     print(second_testSplit[1:5])


# Cut set into bins based on median_income
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])

# Split set by median_income category
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n,strat_test_set_n])
strat_train_set, strat_test_set = strat_splits[0]

print(strat_test_set.describe())
print(strat_train_set.describe()) 
