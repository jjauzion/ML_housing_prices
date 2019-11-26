from sklearn import linear_model
import pandas as pd


def fit(X_train, y_train, X_test, y_name="Prediction"):
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_test = reg.predict(X_test)
    return reg, pd.DataFrame(data=y_test, index=X_test.index, columns=[y_name])
