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
    # trainSplit, testSplit = train_test_split(housing, test_size=0.2, random_state=42)
    # print("First Random Training Sample (Rows 1-5)")
    # print(trainSplit[1:5])
    # print("First Random Test Sample (Rows 1-5)")
    # print(testSplit[1:5])
    # second_trainSplit, second_testSplit = train_test_split(housing, test_size=0.2, random_state=42)
    # print("Second Random Training Sample (Rows 1-5)")
    # print(second_trainSplit[1:5])
    # print("Second Random Test Sample (Rows 1-5)")
    # print(second_testSplit[1:5])


# Split data uniformly into 5 categories, plot on graph
housing["income_cat"] = pd.qcut(housing["median_income"], q=5, labels=[1,2,3,4,5])
    # housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    # plt.xlabel("Income category")
    # plt.ylabel("Number of districts")
    # plt.show()


# Create train and split sets by median_income category
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
# Print ratio of category count / set count
    # print(strat_test_set["income_cat"].value_counts().sort_index() / len(strat_test_set))

# Plot the test dataset to visualize the geographical information
strat_test_set.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                    s=strat_test_set["population"] /100, label="population",
                    c="median_house_value", cmap="jet", colorbar=True,
                    legend=True, sharex=False, figsize=(10,7))
    # plt.show()

# Calculate correlations
corr_matrix = housing.corr()
print(corr_matrix["median_housing_value"].sort_values(ascending=False))