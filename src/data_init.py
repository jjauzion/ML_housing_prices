import csv
import numpy as np
import src.hyper_parameters as hp
import sklearn.preprocessing as sklpre

def     get_from_csv(file2read):
    """
    
    Read a csv file and return a a tuple made of:
    - A list containing the header
    - A list containing the data as follow:
          Each line is an element of the list and 
          for each element of the list each column is an element:
              [[c1, c2, c3], [c1, c2, c3]]

    """
    with open(file2read, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
    with open(file2read, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data = []
        for row in csvreader:
            data.append(row)
    return header, data

def     print_NA(header ,data):

    for i in range(len(header)):
        count = sum([1 for row in data if row[i] == "NA"])
        print("{}.{} : {} NA".format(i, header[i], count))

def     clean_data(header, data):
    header, data = get_from_csv('dataset/train.csv')
    notNumColumn = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 28, 27, 29, 30, 31, 32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]
    for row in data:
        for column in sorted(notNumColumn, reverse=True):
            row.pop(column)
    for column in sorted(notNumColumn, reverse=True):
            header.pop(column)
    #print_NA(header, data)
    column_NA = [2, 8, 25]
    for row in data:
        for column in sorted(column_NA, reverse=True):
            row.pop(column)
    for column in sorted(column_NA, reverse=True):
            header.pop(column)
    strongly_corr_var = [11, 24]
    for row in data:
        for column in sorted(strongly_corr_var, reverse=True):
            row.pop(column)
    for column in sorted(strongly_corr_var, reverse=True):
            header.pop(column)
    for row in data:
        row[:] = [int(column) for column in row]
    return header, data

def     mean_norm_1d(lst):
    if lst.std() == 0:
        return lst
    return (lst - lst.mean()) / lst.std()

def     mean_normalisation(data):
    return np.apply_along_axis(mean_norm_1d, axis=0, arr=data)

def     split_data(data, shuffle="yes"):
    """Split data set between training, cross validation and test set."""
    if shuffle == "yes":
        np.random.shuffle(data)
    m = np.size(data, 0)
    train_index = m * hp.data_split[0] // 100
    cv_index = train_index + m * hp.data_split[1] // 100
    indexes = [train_index, cv_index]
    return tuple(np.split(data, indexes, axis=0))

def     add_poly_features(data, degree=2, start=0, feature_axis=1):
    """Add polynomial features to the data set"""
    poly = sklpre.PolynomialFeatures(degree)
    enriched_data = poly.fit_transform(data[:,start:])
    return np.concatenate((data[:,:start], enriched_data), axis=feature_axis)
