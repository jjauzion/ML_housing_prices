from sklearn import linear_model


def fit(X_train, y_train, X_test):
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_test = reg.predict(X_test)
    return reg, y_test
