# -*-coding:Utf-8 -*

import src.data_init as data_init
import src.linear_regression as lr
import numpy as np
import matplotlib.pyplot as plt
import src.hyper_parameters as hp
import src.learning_curve as lc
import pandas as pd

def     print_data_sample(header, data):
    for i in range(len(header)):
        #print("{}.{} : {}".format(i, header[i], " / ".join([row[i] for row in data[:] if row[71] != '0'])))
        print("{}.{} : {}".format(i, header[i], " / ".join([str(row[i]) for row in data[:5]])))

def     print_data(header, data):
    data_sorted = sorted(data, key=lambda price: price[33])
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('Price ($)', color=color)
    plt.plot([int(row[33]) for row in data_sorted], color=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax1.set_ylabel('Nb of cars', color=color)
    plt.plot([int(row[22]) for row in data_sorted], color=color)
    fig.tight_layout()
    plt.show()

header, data = data_init.get_from_csv('dataset/train.csv')
header, data = data_init.clean_data(header, data)
print_data_sample(header, data)
data = np.array(data)
train_data, val_data, test_data = data_init.split_data(data) 

X, Y = tuple(np.split(train_data, [-1], axis=1))
X = data_init.mean_normalisation(X)
X = np.insert(X, 0, np.ones((1, np.size(X, 0))), axis=1) #add ones for bias
Xval, Yval = tuple(np.split(val_data, [-1], axis=1))
Xval = data_init.mean_normalisation(Xval)
Xval = np.insert(Xval, 0, np.ones((1, np.size(Xval, 0))), axis=1) #add ones for bias

print("Training linear regression...")
theta, cost, conver_iter = lr.train_linear_regression(X, Y, hp.alpha, opt="cost")
if conver_iter < 0:
    print("WARNING: convergence not reached! Consider increasing the nb of iteration")
else:
    print("convergence reached at iter {} ; cost : {}".format(conver_iter, cost))
H = X.dot(theta)
diff_tr = pd.DataFrame(np.absolute(Y - H))
print("diff_tr:\n", diff_tr.describe())
H = Xval.dot(theta)
diff_val = pd.DataFrame(np.absolute(Yval - H))
print("diff_val:\n", diff_val.describe())

print("generating learning curve...")
lc.learning_curve(X, Y, Xval, Yval, hp.alpha)
print("Done!")