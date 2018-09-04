import src.hyper_parameters as hp

def     cost_function(H, theta, X, Y):
    m, n = Y.shape
    cost = 1 / (2 * m) * ((H - Y).T.dot((H - Y))) + hp.regul * theta[1:].T.dot(theta[1:])
    return cost[0, 0]

def     gradient_descent(H, theta, X, Y):
    m, n = Y.shape
    return theta * (1 - hp.alpha * hp.regul / m) - hp.alpha / m * X.T.dot((H - Y))