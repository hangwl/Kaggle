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

