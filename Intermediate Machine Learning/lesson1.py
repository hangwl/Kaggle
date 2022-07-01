from pyexpat import model
import pandas as pd
from sklearn.model_selection import train_test_split

train_path = r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\train.csv"
train_data = pd.read_csv(train_path)

test_path = r"C:\Users\HangWei\OneDrive\Desktop\VSC\Kaggle\Data Sources\Housing_Prices_Competition\test.csv"
test_data = pd.read_csv(test_path)

y = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features].copy()
X_test = test_data[features].copy()

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#print(X_train.head())

from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

def score_model(model, X_t=X_train, X_v=X_val, y_t=y_train, y_v=y_val):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(len(models)):
    mae = score_model(models[i])
    print(f"Model {i+1} MAE: {mae}")

best_model = model_3
best_model.fit(X, y)
test_preds = best_model.predict(X_test)

import os
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': test_preds})
file_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_dir, 'test_predictions.csv')
output.to_csv(file_path, index=False)