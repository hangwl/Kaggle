# Importing relevant libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""
First, we will train a Random Forest Model using the training data set.

The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. 
"""

# Load data and separate target
iowa_file_path = r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv"
home_data = pd.read_csv(iowa_file_path)

#print(home_data.columns) # View data columns

# Select SalePrice as target
y = home_data.SalePrice

# Select Features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

#print(X.describe()) # Summarize Features Data
#print(X.head()) # Preview Top Rows of Features Data

# Split validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a Random Forest Model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y) # Fit model using training data
rf_val_predictions = rf_model.predict(val_X) # Make predictions using validation features data
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y) # Assess Mean Absolute Error by comparing predictions with real y values

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

"""
Now, we train the model on the full training data...
"""

rf_model_full = RandomForestRegressor(random_state=1)
rf_model_full.fit(X, y)
rf_model_full_predictions = rf_model_full.predict(X)
rf_model_full_mae = mean_absolute_error(rf_model_full_predictions, y)

print("MAE for Random Forest Model: {:,.0f}".format(rf_model_full_mae))


"""
Lastly, we apply model to make predictions on test data
"""

test_data_path = r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv"
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

test_preds = rf_model_full.predict(test_X)


# Generate CSV file with predictions

import os
file_dir = os.path.dirname(os.path.abspath(__file__))

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

file_path = os.path.join(file_dir, 'test_predictions.csv')

output.to_csv(file_path, index=False)


