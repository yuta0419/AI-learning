import pandas as pd
from sklearn import datasets

housing = datasets.fetch_california_housing()
housing_df = pd.DataFrame(housing.data,columns=housing.feature_names)
housing_df["PRICE"] = housing.target
print(housing_df.head())