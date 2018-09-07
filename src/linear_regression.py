import src.hyper_parameters as hp
import numpy as np
import matplotlib.pyplot as plt
import sys

def     cost_function(H, theta, Y, regul=0):
    m, n = Y.shape
    cost = 1 / (2 * m) * ((H - Y).T.dot((H - Y))) + regul / (2 * m) * theta[1:].T.dot(theta[1:])
    if np.isnan(cost).any():
        print("Divergence -> iteration interupted. Consider trying with a lower learning rate")
        sys.exit()
    return cost[0, 0]

def     gradient_descent(H, theta, X, Y, alpha):
    m, n = Y.shape
    return theta * (1 - hp.alpha * hp.regul / m) - alpha / m * X.T.dot((H - Y))

def     plot_convergence(cost):
    x = np.arange(hp.nb_iteration)
    plt.plot(x, cost)
    plt.show()

def     train_linear_regression(X, Y, alpha, regul=0, opt=""):
    theta = np.zeros((np.size(X, 1), 1))
    if opt == "cost":
        cost = np.zeros(hp.nb_iteration)
    check_conv = 0
    convergence_iter = -1
    for i in range(hp.nb_iteration):
        hypothesis = X.dot(theta)
        if opt == "cost":
            cost[i] = cost_function(hypothesis, theta, Y, regul=regul)
            check_conv = abs(cost[i] - (cost[i - 1] if i > 0 else 0))
        else:
            check_conv = abs(cost_function(hypothesis, theta, Y, regul=regul) - check_conv)
        theta = gradient_descent(hypothesis, theta, X, Y, alpha)
        if check_conv < 0.001 and convergence_iter < 0:
            convergence_iter = i
    if opt == "cost":
        plot_convergence(cost)
    else:
        cost = cost_function(hypothesis, theta, Y, regul=regul)
    return theta, cost, convergence_iter