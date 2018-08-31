import csv
import numpy as np

def     get_data_from_csv(file2read):
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
        nb_line = sum(1 for row in csvreader)
    with open(file2read, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        nb_column = len(header)
        data = []
        for row in csvreader:
            data.append(row)
    return header, data
