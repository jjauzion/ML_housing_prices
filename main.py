# -*-coding:Utf-8 -*

from src.get_data import get_data_from_csv
import numpy as np

data = get_data_from_csv('dataset/train.csv')
print(data)
