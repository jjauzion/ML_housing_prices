import csv
import numpy as np

def     get_data_from_csv(file2read):
    """
    
    Read a csv file and return a list containing the data
    Each line is an element of the list and for each element of the list each column is an element:
    [[c1, c2, c3], [c1, c2, c3]]

    """
    with open(file2read, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        nb_line = sum(1 for row in csvreader)
    with open(file2read, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        row = next(csvreader)
        nb_column = len(row)
        data = []
        for row in csvreader:
            data.append(row)
    return data
