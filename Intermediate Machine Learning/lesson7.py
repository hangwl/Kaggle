# Data Leakage

"""
Data leakage can ruin models in subtle and dangerous ways.
It happens when the training data contains information about the target, but similar data will not be available when the model is used for prediction.
This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.
i.e. leakage causes a model to be accurate only until the before the decision making phase.

2 main types of leakage: 

    1. Target leakage 

    Occurs when your predictors include data that will not be available at the time you make predictions.
    Important to think about target leakage in terms of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions.
    To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.
    e.g. if sick == True, there is high frequency that patient took medicine recently column value changes

    2. Train-Test Contamination

    Recall that validation is meant to be a measure of how the model does on data that is hasn't considered before.
    The process can be corrupted in subtle ways if the validation data affects the preprocessing behaviour.
    To prevent this, if our validation is based on a simple train_test_split, exclude the validation from any type of fitting, including the fitting of preprocessing steps.
    eg. preprocessing (using a imputer) before calling train_test_split

We want to identify potential leaks and drop them.

Refer to https://www.kaggle.com/code/alexisbcook/data-leakage for examples
"""

