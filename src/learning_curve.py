import numpy as np
import src.linear_regression as lr
import matplotlib.pyplot as plt

def     learning_curve(X, Y, Xval, Yval, alpha, regul=0):
    m = np.size(Y, axis=0)
    iterator = list(range(10, m, 10))
    error_train = np.zeros(len(iterator))
    error_cv = np.zeros(len(iterator))
    for i, index in enumerate(iterator):
        sub_X = X[:index]
        sub_Y = Y[:index]
        theta, cost, conv_iter = lr.train_linear_regression(sub_X, sub_Y, alpha)
        hypothesis = sub_X.dot(theta)
        error_train[i] = lr.cost_function(hypothesis, theta, sub_Y, regul=0)
        hypothesis = Xval.dot(theta)
        error_cv[i] = lr.cost_function(hypothesis, theta, Yval, regul=0)
    plt.plot(iterator, error_train, color='red', label='train error')
    plt.plot(iterator, error_cv, color='green', label='cross validation error')
    plt.legend()
    plt.xlabel('nb of training exemples')
    plt.ylabel('error')
    plt.show()