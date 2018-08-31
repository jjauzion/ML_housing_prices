import csv
import numpy as np

def     read_csv(file2read):
    with open(file2read, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        nb_line = sum(1 for row in csvreader)
        print("nb ligne = ", nb_line)
        row = next(csvreader)
        print("1st ligne : |{}|", row)
        for row in csvreader:
            #print(', '.join(row))
            nb_line += 1
    print("\n-------------------------------\n")
