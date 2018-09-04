# -*-coding:Utf-8 -*

import src.data_init as data_init
import src.linear_regression as lr
import numpy as np
import matplotlib.pyplot as plt
import src.hyper_parameters as hp

def     print_data_sample(header, data):
    for i in range(len(header)):
        #print("{}.{} : {}".format(i, header[i], " / ".join([row[i] for row in data[:] if row[71] != '0'])))
        print("{}.{} : {}".format(i, header[i], " / ".join([str(row[i]) for row in data[:10]])))

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
#print_data(header, data)
data = np.array(data)
X, Y = tuple(np.split(data, [-1], axis=1))
X = data_init.mean_normalisation(data)
X = np.insert(X, 0, np.ones((1, np.size(X, 0))), axis=1) #add ones for bias
theta = np.zeros((np.size(X, 1), 1))
cost = np.zeros((1, hp.nb_iteration))
print("iterating...")
for i in range(hp.nb_iteration):
    hypothesis = X.dot(theta)
    cost[0, i] = lr.cost_function(hypothesis, theta, X, Y)
    theta = lr.gradient_descent(hypothesis, theta, X, Y)
print("iteration completed!")
print("cost : ", cost)
hypothesis = X.dot(theta)
print(np.concatenate((Y, hypothesis), axis=1))