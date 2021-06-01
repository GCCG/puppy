""" import sys
import copy
import numpy as np


class Yes():
    def __init__(self):
        self.a = 1
        self.b = 2
    
    def __str__(self):
        return str(self.a) + ':' + str(self.b)

    def __eq__(self, other):
        return self.a ==  other.a

if __name__ == "__main__":
    # y = ['a','b','c','d']
    # x = np.random.randint(0,4,3)
    # z= []
    # for i in x:
    #     z.append(y[i])
    # print("y: ", y)
    # print("x: ", x)
    # print("z: ", z)
    
    beta_h = abs(np.random.rand(10))*3
    sigma_h = abs(np. random.rand(10))

    beta_h_s = min(beta_h)*np.ones(10)
    u_r = min(beta_h)/beta_h


    # example1 = Yes()
    # example2 = Yes()
    # print(example1==example2)
    # x = [example1, example2]
    # b = list(x)
    # c = copy.deepcopy(x)
    # print(x[0], b[0], c[0])
    # b[0].a = 3
    # print(x[0], b[0], c[0])
    # c[0].a = 10
    # print(x[0], b[0], c[0])
    # v = {}
    # try:
    #     print(v['Tom'][0])
    # except :
    #     v['Tom'] = [1]
    #     print(v['Tom'][0])
 # look at here
    
 """

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

def func():
    return 3,2

if __name__=="__main__":
    x = func()
    print(x)
    x,y = func()
    print(x,y)