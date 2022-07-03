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
from sklearn import preprocessing
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

#print(X_train.head())

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for Numerical Data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for Categorical Data
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant')), #'constant' yields slightly better MAE than 'most_frequent' in this case
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Define Model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in pipeline
my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

# Preprocessing of Training Data, Fit Model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_val)

print('MAE:', mean_absolute_error(y_val, preds))

# Preprocessing of Test Data, Fit Model
preds_test = my_pipeline.predict(X_test)

import os
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
file_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_dir, 'lesson_4_test_predictions.csv')
output.to_csv(file_path, index=False)