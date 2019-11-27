from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import math
import numpy as np


def rmsle_score(y_true, y_predict, zero_truncature=False, verbosity=1):
    """
    Compute the Root Mean Square Ln Error between y_true and y_pred
    :param y_true:              [pandas DataFrame]
    :param y_predict:           [pandas DataFrame]
    :param zero_truncature:     [Bool] If true, all negative value will be set to 0. If False, an error will be raise
                                in case of negative value.
    :return:                    [float] rmsle value
    """
    if zero_truncature:
        if verbosity > 0:
            if y_true[y_true < 0].count()[0] > 0 or y_predict[y_predict < 0].count()[0]:
                print(f'WARNING: negative values found in the data. Negative value will be set to 0 to compute log.')
                print(f'y_pred = \n{y_predict.sort_values(by=y_predict.columns[0])[:10]}')
        y_true = y_true.mask(y_true.iloc[:, 0] < 0, other=0)
        y_predict = y_predict.mask(y_predict.iloc[:, 0] < 0, other=0)
    msle = metrics.mean_squared_log_error(y_true, y_predict)
    return np.sqrt(msle)
    # return np.sqrt(metrics.mean_squared_log_error(y_true, y_predict))


def fit_predict(X_train, y_train, X_test, model="LinearReg", seed=None, verbosity=1, **kwargs):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param model:       [str | object] string among ["LinearReg"]
                        or any model object that implement fit and predict method
    :param seed:        [int] seed value
    :param verbosity:   [int] verbosity level
    :return:
    """
    if model == "LinearReg":
        reg = linear_model.LinearRegression()
    elif model == "Ridge":
        reg = linear_model.Ridge(alpha=kwargs["alpha"] if "alpha" in kwargs else 0.5, random_state=seed)
    else:
        reg = model
    model_name = model if isinstance(model, str) else reg.__class__.__name__
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)
    if verbosity > 0:
        print(f'Training completed!')
        train_score = reg.score(X_train, y_train)
        print(f'Train score (coef of determination R^2) = {train_score}')
    return reg, pd.DataFrame(data=y_test_pred, index=X_test.index, columns=[model_name])
