# XGBoost

"""
AKA Extreme Gradient Boosting

With gradient boosting, we can build and optimize models to achieve state-of-the-art results on a variety of datasets.

Gradiant boosting is a method that goes through cycles to iteratively add models into an ensemble.

It begins by initializing the ensemble with a single model, whose predictions can be pretty naive.
(Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

Then, we start the iterative cycle - Make predictions > Calculate loss > Train new model > Add new model to ensemble > Make predictions

Full Steps:

1. Load training and validation data in X_train, X_valid, y_train, y_valid

2. Import the scikit-learn API for XGBoost (xgboost.XGBRegressor)
    Consider Parameter Tuning 
        - my_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4) 
        # n_estimators - too low leads to underfitting and general inaccuracy & too high leads to overfitting on training data and inaccuracy in testing data
        # default learning_rate is 0.1
        # n_jobs utilizes parallel processing on no. of cores which can be useful to process larger datasets
        - my_model.fit(..., early_stopping_rounds=5)
        # early_stopping_rounds offers a way to automatically find the ideal value for n_estimators
        # i.e. we stopp after 5 straight rounds of deteriorating validation scores

3. Make predictions and evaluate the model (see MAE)


"""

# Step 0: Prepare data

import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv")
X_test_full = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv")

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == "object"]

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Step 1: Build Model

from xgboost import XGBRegressor

# Define the model
my_model_1 = XGBRegressor(random_state=0)

# Fit the model
my_model_1.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = predictions_1 = my_model_1.predict(X_valid)

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid)
print("Mean Absolute Error:" , mae_1)

## After we have trained a default model as a baseline, we can adjust parameters to optimize performance.

# Step 2: Improve the Model

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model_2.fit(X_train, y_train)
predictions_2 = my_model_2.predict(X_valid)
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:" , mae_2)

