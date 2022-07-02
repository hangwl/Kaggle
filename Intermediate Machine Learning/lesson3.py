# Categorical Variables

# 3 basic approaches:

# 1. Drop categorical variables
# 2. Ordinal Encoding - assign each unique value to a different integer
# 3. One-Hot Encoding - create new columns indicating the presence (or absence) of each possible value in the original data

# In general, approach #3 will typically perform best, and approach #1 the worst, but it varies on a case-by-case basis

# In this exercise, we will continue to use the Housing Prices data

# load relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# load training and test data
X = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv", index_col="Id")
X_test = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv", index_col="Id")

# Remove rows with missing target, and separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print(X_train.head()) # Preview features

# We notice that the dataset contains both numerical and categorical variables. We will need to encode the categorical data before training a model.

# To compare different models, we will use the same score_dataset() function from the tutorial. This function reports the MAE from a random forest model

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Approach 1 - Dropping Cols with Categorical Data
X_train_drop = X_train.select_dtypes(exclude=['object'])
X_val_drop = X_val.select_dtypes(exclude=['object'])

print(f'MAE from Approach 1: {score_dataset(X_train_drop, X_val_drop, y_train, y_val)}')

# Approach 2 - Ordinal Encoding

# Lets take a look at the 'Condition2' column.
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_val['Condition2'].unique())
# Training and validation sets produce entries that are unique in each set

# The problem with this is that if we fit an ordinal encoder to the training set, it may exclude values from the validation set, and the code will throw an error.

# A simple approach is to drop the problematic categorical columns...

# Categorical Columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if set(X_val[col]).issubset(set(X_train[col]))]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

print(f"Categorical columns that will be ordinal encoded: {good_label_cols}")
print(f"\nCategorical columns that will be dropped from the dataset: {bad_label_cols}")

from sklearn.preprocessing import OrdinalEncoder

# Drop categorical columns that will not be encoded
X_train_label = X_train.drop(bad_label_cols, axis=1)
X_val_label =X_val.drop(bad_label_cols, axis=1)

# Apply ordinal encoder
ordinal_encoder = OrdinalEncoder()
X_train_label[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
X_val_label[good_label_cols] = ordinal_encoder.transform(X_val[good_label_cols])

print(f"MAE from Approach 2 (Ordinal Encoding): {score_dataset(X_train_label, X_val_label, y_train, y_val)}")

# Before we try Approach 3, need to investigate cardinality

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
print(sorted(d.items(), key=lambda x: x[1]))

# The output shows the number of unique values in each respective column with categorical data
# We refer to the number of unique entries as the cardinality of that categorical variable.