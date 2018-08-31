# -*-coding:Utf-8 -*

from src.get_data import get_data_from_csv
import numpy as np
import matplotlib.pyplot as plt

def     print_data_sample(header, data):
    for i in range(len(header)):
        #print("{}.{} : {}".format(i, header[i], " / ".join([row[i] for row in data[:] if row[71] != '0'])))
        print("{}.{} : {}".format(i, header[i], " / ".join([row[i] for row in data[:10]])))

def     print_data(header, data):
    data_sorted = sorted(data, key=lambda price: price[80])
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('Price ($)', color=color)
    plt.plot([int(row[80]) for row in data_sorted], color=color)
    """
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax1.set_ylabel('Pool area (feet**2)', color=color)
    plt.plot([int(row[71]) for row in data_sorted], color=color)
    fig.tight_layout()
    """
    plt.show()

header, data = get_data_from_csv('dataset/train.csv')
print_data_sample(header, data)
print_data(header, data)
