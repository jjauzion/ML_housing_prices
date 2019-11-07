# -*-coding:Utf-8 -*

import src.data_init as data_init
import src.linear_regression as lr
import numpy as np
import matplotlib.pyplot as plt
import src.hyper_parameters as hp
import src.learning_curve as lc
import pandas as pd
import sys

def     print_data_sample(header, data):
    for i in range(len(header)):
        #print("{}.{} : {}".format(i, header[i], " / ".join([row[i] for row in data[:] if row[71] != '0'])))
        print("{}.{} : {}".format(i, header[i], " / ".join([str(row[i]) for row in data[:5]])))

header, data = data_init.get_from_csv('dataset/train.csv')
header, data = data_init.clean_data(header, data)
print_data_sample(header, data)
data = np.array(data)
data = data_init.add_poly_features(data, degree=1, start=1, feature_axis=1) #also add ones in 1st column for bias
train_data, val_data, test_data = data_init.split_data(data, shuffle="no")

id_tr, Xtraining, Ytraining = tuple(np.split(train_data, [1, -1], axis=1))
Xtraining = data_init.mean_normalisation(Xtraining)
#Xtraining = np.insert(Xtraining, 0, np.ones((1, np.size(Xtraining, 0))), axis=1) #add ones for bias
print("Training linear regression...")
theta, cost, conver_iter = lr.train_linear_regression(Xtraining, Ytraining, hp.alpha, opt="cost")
if conver_iter < 0:
    print("WARNING: convergence not reached! Consider increasing the nb of iteration")
else:
    print("convergence reached at iter {} ; cost : {}".format(conver_iter, cost))
H = Xtraining.dot(theta)
diff_tr = pd.DataFrame(np.absolute(Ytraining - H))
print("diff_tr:\n", diff_tr.describe())
sys.exit()

id_val, Xval, Yval = tuple(np.split(val_data, [1, -1], axis=1))
Xval = data_init.mean_normalisation(Xval)
#Xval = np.insert(Xval, 0, np.ones((1, np.size(Xval, 0))), axis=1) #add ones for bias
H = Xval.dot(theta)
diff_val = pd.DataFrame(np.absolute(Yval - H))
print("diff_val:\n", diff_val.describe())

print("generating learning curve...")
lc.learning_curve(Xtraining, Ytraining, Xval, Yval, hp.alpha)
print("Done!")