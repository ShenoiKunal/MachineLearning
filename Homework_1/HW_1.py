from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# Load data into PD dataframe
housing = pd.read_csv(Path("datasets/housing.csv"))

# Show that the random samples are always the same with the same random_state
print("Random Data and Random State\n--------------------")
trainSplit, testSplit = train_test_split(housing, test_size=0.2, random_state=42)
print("First Random Training Sample (Rows 1-5)")
print(trainSplit[1:5])
print("First Random Test Sample (Rows 1-5)")
print(testSplit[1:5])
second_trainSplit, second_testSplit = train_test_split(housing, test_size=0.2, random_state=42)
print("Second Random Training Sample (Rows 1-5)")
print(second_trainSplit[1:5])
print("Second Random Test Sample (Rows 1-5)")
print(second_testSplit[1:5])

# Split data uniformly into 5 categories, plot on graph
housing["income_cat"] = pd.qcut(housing["median_income"], q=5, labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.title("Number of Districts vs Income Category")
plt.show()

# Create train and split sets by median_income category
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"],
                                                   random_state=42)

# Plot the test dataset to visualize the geographical information
strat_test_set.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                    s=strat_test_set["population"] / 100, label="population",
                    c="median_house_value", cmap="jet", colorbar=True,
                    legend=True, sharex=False, figsize=(10, 7))
plt.title("Geographical Information based on Population highlighted by Median House Value")
plt.show()

# Plot the whole dataset for Median House Value vs Median Income
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.title("Median House Value vs Median Income (Raw)")
plt.show()

# Remove the data that does not align with the pattern
bad_data = housing[(housing['median_house_value'].isin([450000, 350000, 280000])) | (housing['median_house_value'] >= 499500)].index

# Print indices to drop
print("\n\nData to be dropped\n--------------------")
print(bad_data)
cleaned_housing = housing.drop(bad_data)

# Re-plot Median House Value vs Median Income for cleaned dataset
cleaned_housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.title("Median House Value vs Median Income (Cleaned)")
plt.show()

# Calculate correlations
print("\n\nCorrelations\n--------------------")
corr_matrix = strat_train_set.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Prepare data for machine learning
    # Create fresh sets
train = strat_train_set.drop("median_house_value", axis=1)
train_labels = strat_train_set["median_house_value"].copy()
    # Impute missing data
imputer = SimpleImputer(strategy="median")
train_numeric_only = train.select_dtypes(include=[np.number])
imputer.fit(train_numeric_only)
    # Validate Medians are correct
print("\n\nValidate Medians for imputing\n--------------------")
print(f"Imputer Statistics: {imputer.statistics_}")
print(f"Actual: {train_numeric_only.median().values}")
    # Adjusted train set without null values
train_imputed = imputer.transform(train_numeric_only)

# Perform train using linear regression
lin_reg = LinearRegression()
lin_reg.fit(train_imputed, train_labels)
train_predictions = lin_reg.predict(train_imputed)
    # Print top 5 predictions
print("\n\nTraining Predictions and RMSE\n--------------------")
print(train_predictions[:5].round(-2))
print(train_labels[:5].values)

# Calculate RMSE
rmse = root_mean_squared_error(train_labels, train_predictions)
print (f"RMSE: {rmse}")

# Perform test using linear regression
    # Remove desired value
test = strat_test_set.drop("median_house_value", axis=1)
    # Copy of actual values
test_labels = strat_test_set["median_house_value"].copy()
    # Use only numeric data
test_num = test.select_dtypes(include=[np.number])
    # Replace null values with median
test_imputed = imputer.transform(test_num)
    # Predict using model fit on training data
predictions = lin_reg.predict(test_imputed)
    # Print top 5 predictions
print("\n\nTest Predictions and RMSE (same model)\n--------------------")
print(f"Predictions: {predictions[:5].round(-2)}")
print(f"Actual: {test_labels[:5].values}")

# Calculate RMSE
rmse = root_mean_squared_error(test_labels, predictions)
print (f"RMSE: {rmse}")