# Lesson 2: Missing Values

# There are 3 basic approaches to handling missing values
# Do consider information loss when dropping data

# 1. Dropping columns with missing values
# 2. Imputation - fill in missing values with column mean
# 3. An extension to imputation - add additional column to mark data with missing row entries


import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


train_data = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv", index_col='Id')
test_data = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv", index_col='Id')

# Drop rows with missing SalePrice
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # refer to documentation https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html

# Define y
y = train_data.SalePrice
# Define X
train_data.drop(['SalePrice'], axis = 1, inplace=True)
# Use only numerical predictors to keep things simple
X = train_data.select_dtypes(exclude=['object'])
# Drop object data fom test_data
test_data_numerical = test_data.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preliminary Investigation

# Shape of training data (num_rows, num_columns)
print(X_train.shape)
# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Function for comparing different approaches
def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

# Drop columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

X_train_red =  X_train.drop(cols_with_missing, axis=1)
X_val_red = X_val.drop(cols_with_missing, axis=1)

print("MAE (Drop cols with missing values):", score_dataset(X_train_red, X_val_red, y_train, y_val))

# Imputation
my_imputer = SimpleImputer() # default strategy = mean
X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train)) # Use fit_transform only training data and transform on validation data
X_val_imputed = pd.DataFrame(my_imputer.transform(X_val))
# The process of imputation removed column names, need to replace...
X_train_imputed.columns = X_train.columns
X_val_imputed.columns = X_val.columns

print("MAE (mean Imputation):", score_dataset(X_train_imputed, X_val_imputed, y_train, y_val))

# Generate Test Predictions via median imputation strategy
final_imputer = SimpleImputer(strategy='median')
X_train_final = pd.DataFrame(final_imputer.fit_transform(X_train))
X_val_final = pd.DataFrame(final_imputer.fit_transform(X_val))
X_train_final.columns = X_train.columns
X_val_final.columns = X_val.columns

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train_final, y_train)

# Get validation predictions and MAE
preds_val = model.predict(X_val_final)
print("MAE (median Imputation):", mean_absolute_error( y_val, preds_val))

# Preprocess Test Data
X_test_final = pd.DataFrame(final_imputer.transform(test_data_numerical))
X_test_final.columns = test_data_numerical.columns

# Get Test Predictions
test_preds = model.predict(X_test_final)

import os
output = pd.DataFrame({'Id': X_test_final.index,
                       'SalePrice': test_preds})
file_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_dir, 'lesson_2_test_predictions.csv')
output.to_csv(file_path, index=False)