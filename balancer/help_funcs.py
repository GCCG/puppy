from sympy import *
import numpy as np

def create_symbol_list(prefix, row, column):
    x = []
    for i in range(row):
        for j in range(column):
            x.append(symbols(prefix+str(row)+str(column)))