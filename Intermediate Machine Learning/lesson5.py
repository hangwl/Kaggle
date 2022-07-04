# Cross-Validation

"""
Machine learning is an iterative process and choices need to be made as to what predictive variables to use, what arguments to supply those models, etc.
So far in previous lessons, these choices have been made in a data-driven way by measuring model quality with a validation (or holdout) set.

But there are some drawbacks to this approach.

Imagine a dataset with 5000 rows and typically, 20% of the data will be used as the validation set.
However, this leaves some random chance in that a model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows.

In general, the larger the validation set, the less randomness (or noise) there is in our measure of quality, and the more reliable it will be.
Unfortunately, we can only get a large validation set by removing rows from our training data, and smaller datasets mean worse models...

In cross-validation, we run our modeling process on different subsets of data to get multiple measures of model quality.

That is, we could divide the data into 5 folds, and iteratively run the modeling process on each validation fold.

However, cross-validation will require deliberation for a longer processing time.

"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv", index_col='Id')
test_data = pd.read_csv(r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv", index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

# print(X.head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())
# Step 1 - Write a get_score function that reports the average (over 3 cross-validation folds) MAE
def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

# Step 2 - Test different parameter values
results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

import matplotlib.pyplot as plt
#%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()

# Step 3 - Find the best parameter value
n_estimators_best = n_estimators_best = min(results, key=results.get)
