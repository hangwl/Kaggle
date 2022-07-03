# Pipelines

"""
Pipelines are a simple way to keep data preprocessing and modeling code organized.
Specifically, a pipeline bundles preprocessing and modeling steps so I can use the whole bundle as if it were a single step.
Pipelines have benefits that include:
1. Cleaner code - Accounting for data at each step of preprocessing can get messy. With a pipeline, no need to manually keep track of training and validation data at each step.
2. Fewer bugs - Fewer opportunities to misapply a step or forget a preprocessing step.
3. Easier to productionize - It can be surprisingly hard to transition a model from prototype to something deployable at scale.
4. More options for model validation -
"""

import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv", index_col='Id')
X_test = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv", index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Select columns with relatively low cardinality
categorical_cols = [col for col in X_train.columns if X_train[col].nunique() < 10 and X_train[col].dtype == 'object']

# Select numerical columns
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
X_val = X_val[my_cols].copy()
X_test = X_test[my_cols].copy()

print(X_train.head())