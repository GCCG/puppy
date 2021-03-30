import numpy as np
# from function import Function   #定义法求导工具
# from lagb import *  #线性代数工具库
from scipy.linalg import *
# import cvxpy as cp

n=3 #x的长度
mu=2
lj=np.ones(1)   #λj初值，长度等于等式限制条件个数
li=np.ones(6)   #λi初值，长度等于不等式限制条件个数

def func(x):
    r = cp.square(x[0]-x[1])
    return #函数

def hj(x):
    #构造数组h，第j位是第j+1个等式限制条件计算的值
    h = [x[0] + x[1] - 1]
    return h

def gi(x):
    #构造数组g，第i位是第i+1个等式限制条件计算的值
    g = [x[0] - x[1] - 1]
    return g

def myFunc(x):
    return  func(x)+\
        1/(2*mu)*np.sum(np.power(np.where(li-mu*gi(x)>0,li-mu*gi(x),0),2)-np.power(li,2))-\
            np.sum(lj*hj(x))+0.5*mu*np.sum(np.power(hj(x),2))

def cdt(x):
    return np.sqrt(np.sum(np.power(hj(x),2)))+np.sqrt(np.sum(np.power(np.where(gi(x)>0,0,gi(x)),2)))

sigma2=1.5  #放大因子
e2=0.001
beta=0.5
x=np.array([2.0,2.0,2.0])   #初值点
k1=0
while cdt(x)>=e2:
    e=0.001
    beta1=1
    sigma=0.4
    rho=0.55
    tar=myFunc
    k=0
    # d=-tar.grad(x)
    d = - np.gradient(tar)
    x1=x
    while tar.norm(x)>e:
        a=1
        if not (tar.value(x+a*d)<=tar.value(x)+rho*a*dot(turn(tar.grad(x)),d) and \
            np.abs(dot(turn(tar.grad(x+a*d)),d))>=sigma*dot(turn(tar.grad(x)),d)):
            a=beta1
            while tar.value(x+a*d)>tar.value(x)+rho*a*dot(turn(tar.grad(x)),d):
                a*=rho
            while np.abs(dot(turn(tar.grad(x+a*d)),d))<sigma*dot(turn(tar.grad(x)),d):
                a1=a/rho
                da=a1-a
                while tar.value(x+(a+da)*d)>tar.value(x)+rho*(a+da)*dot(turn(tar.grad(x)),d):
                    da*=rho
                a+=da
        lx=x
        x=x+a*d
        beta=np.max((dot(turn(tar.grad(x)),tar.grad(x)-tar.grad(lx))/(tar.norm(lx)**2),0))  #PRP+
        d=-tar.grad(x)+beta*d
        k+=1
        print(k1,k)
    if cdt(x)/cdt(x1)>beta:
        mu*=sigma2
    k1+=1
print(x)