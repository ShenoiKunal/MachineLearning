from dataclasses import asdict
from pathlib import Path
import pandas as pd

# Load data into PD dataframe
housing = pd.read_csv(Path("datasets/housing.csv"))

print(housing.head())