# Importing relevant libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""
First, we will train a Random Forest Model using the training data set.
"""


# Load data and separate target
iowa_file_path = r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Intro_to_Machine_Learning\home-data-for-ml-course (1)\train.csv"
home_data = pd.read_csv(iowa_file_path)

print(home_data.columns) # View data columns

# Select SalePrice as target
y = home_data.SalePrice

# Select Features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

print(X.describe()) # Summarize Features Data
print(X.head()) # Preview Top Rows of Features Data

# Split validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a Random Forest Model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y) # Fit model using training data
rf_val_predictions = rf_model.predict(val_X) # Make predictions using validation features data
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y) # Assess Mean Absolute Error by comparing predictions with real y values

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))