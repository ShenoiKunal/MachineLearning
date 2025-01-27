from dataclasses import asdict
from pathlib import Path
from pydoc import describe

import pandas as pd

# Load data into PD dataframe
housing = pd.read_csv(Path("datasets/housing.csv"))

print("\nHEAD\n--------\n",housing.head())
print("\nInfo\n--------")
print(housing.info())
print("\nOcean Proximity\n--------\n",housing["ocean_proximity"].value_counts())
print("\nDescribe\n--------\n",housing.describe())