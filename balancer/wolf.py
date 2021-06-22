from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from ..parameters import *
import numpy as np
from scipy.optimize import minimize
import sys
from matplotlib import pyplot as plt
import json
import time
# from brokenaxes import brokenaxes

# Control task model here
VA_COM_MEAN = 0.5
VA_COM_VAR = 1
VA_DATA_MEAN = 1
VA_DATA_VAR = 1
ADJUST = 0.0001

INITIAL_VALUE_NUM = 20

# Control algorithm parameters here
PENALTY = 10000000

DISCOUNT = 0.8


def gen_objective(args, mode=0, argA=None, argB=None):
    server_num, sigma_d, beta_d, sigma_h, beta_h, c, path_bans = args
    mu = np.zeros(server_num)
    omega = np.zeros(server_num)

    for j in range(server_num): 
        mu[j] = c[j]/beta_h
        omega[j] = ((sigma_h/c[j])**2*mu[j]**2+1)/(2*mu[j]) 
    # print("mu\n",mu)
    # print("omega:\n",omega)
    if mode == 0: 
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            # print("mu\n",mu)
            # print("mu - Lam:\n",mu - Lam)
            # print("omega:\n",omega)
            # print("omega[3]*Lam[3]/(mu[3]-Lam[3])\n",omega[2]*Lam[2]/(mu[2]-Lam[2]))
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
                    if i != j and x[server_num**2+i*server_num+j] > 0 and x[i*server_num+j]>0:
                        f = f + x[i*server_num+j]*(beta_d/x[server_num**2+i*server_num+j])
            return f
    elif mode == 1: # 固定卸载决策优化带宽分配 
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + argA[i*server_num + j]
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    f = f + argA[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
                    if i != j and x[i*server_num+j] > 0 and argA[i*server_num+j] > 0:
                        f = f + argA[i*server_num+j]*(beta_d/x[i*server_num+j])
                        # f = f + argA[i*server_num+j]*beta_d/x[i*server_num+j]
            return f
    elif mode == 2:# 固定带宽分配优化卸载决策
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
                    if i != j and argB[i*server_num+j] > 0 and argB[i*server_num+j] > 0:
                        f = f + x[i*server_num+j]*(beta_d/argB[i*server_num+j])
                        # f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
            return f
    elif mode == 3: # 带宽取最小值
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            # print("mu\n",mu)
            # print("mu - Lam:\n",mu - Lam)
            # print("omega:\n",omega)
            # print("omega[3]*Lam[3]/(mu[3]-Lam[3])\n",omega[2]*Lam[2]/(mu[2]-Lam[2]))
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
                    if i != j:
                        f = f + 1 # 路径带宽取最小值，此时无论传输多少，传输延迟都是1
            return f
    elif mode == 4: # 只求计算延迟
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            # print("mu\n",mu)
            # print("mu - Lam:\n",mu - Lam)
            # print("omega:\n",omega)
            # print("omega[3]*Lam[3]/(mu[3]-Lam[3])\n",omega[2]*Lam[2]/(mu[2]-Lam[2]))
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
            return f
    elif mode == 5: # 只求传输延迟
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            # print("mu\n",mu)
            # print("mu - Lam:\n",mu - Lam)
            # print("omega:\n",omega)
            # print("omega[3]*Lam[3]/(mu[3]-Lam[3])\n",omega[2]*Lam[2]/(mu[2]-Lam[2]))
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    if i != j:
                        f = f + x[i*server_num+j]*(beta_d/x[server_num**2+i*server_num+j])
            return f
    elif mode == 6: # 带宽取最大值
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            # print("mu\n",mu)
            # print("mu - Lam:\n",mu - Lam)
            # print("omega:\n",omega)
            # print("omega[3]*Lam[3]/(mu[3]-Lam[3])\n",omega[2]*Lam[2]/(mu[2]-Lam[2]))
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
                    if i != j and x[i*server_num+j]>0 and argB[i*server_num+j] > 0 and argB[i*server_num+j] > 0:
                        f = f + x[i*server_num+j]*beta_d/path_bans[i,j]
            return f
    return func

def gen_deriv_for_obj(args, mode=0, argA=None, argB=None):
    """ 使用了jac后不仅性能变差了，而且收敛速度并没有提升 """
    server_num, sigma_d, beta_d, sigma_h, beta_h, c = args
    mu = np.zeros(server_num)
    omega = np.zeros(server_num)

    for j in range(server_num): 
        mu[j] = c[j]/beta_h
        omega[j] = ((sigma_h/c[j])**2*mu[j]**2+1)/(2*mu[j])

    
    if mode == 0:
        def jac(x):
            jac = np.zeros(len(x))
            lam = lams(x, server_num)
            for i in range(server_num):
                for j in range(server_num):
                    if i !=j and x[server_num**2 + i*server_num + j] > 0:
                        jac[i*server_num + j] = deriv_f(lam=lam[j], mu=mu[j], omega=omega[j]) + beta_d/x[server_num**2 + i*server_num + j]
                        jac[server_num**2 + i*server_num + j] = - x[server_num*i + j]*beta_d/x[server_num**2 + i*server_num + j]**2
            return jac
    elif mode == 1: # 固定卸载决策优化带宽分配 
        def jac(x):
            jac = np.zeros(len(x))
            lam = lams(argA, server_num)
            for i in range(server_num):
                for j in range(server_num):
                    if i != j and x[i*server_num+j] > 0:
                        jac[i*server_num + j] = - argA[server_num*i + j]*beta_d/x[i*server_num + j]**2

            return jac
    elif mode == 2:# 固定带宽分配优化卸载决策
        def jac(x):
            jac = np.zeros(len(x))
            lam = lams(x, server_num)
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    if i != j and argB[i*server_num + j] > 0:
                        jac[i*server_num + j] = deriv_f(lam=lam[j], mu=mu[j], omega=omega[j]) + beta_d/argB[i*server_num + j]
            return jac
    return jac

def deriv_f(lam, mu, omega):
    return omega * mu / (mu - lam)

def lams(x, server_num):
    Lam = np.zeros(server_num)
    for i in range(server_num):
        for j in range(server_num):
            Lam[j] = Lam[j] + x[i*server_num + j]
    return Lam


def gen_args(ng):
    s_list = ng.getServerList()
    server_num = len(s_list)
    sigma_d = VA_DATA_VAR
    beta_d = VA_DATA_MEAN
    sigma_h = VA_COM_VAR
    beta_h = VA_COM_MEAN
    c = np.zeros(len(s_list))
    for j in range(len(s_list)):
        c[j] = s_list[j].getRscAmount()
    return (server_num, sigma_d, beta_d, sigma_h, beta_h, c, ng.getPathBanMatrix())


# Functions for generating, checking and managing  constraints
# Almost all constraints are adjusted, by minusing ADJUST.
def check_constraints(conds, x):
    for i in range(len(conds)):
        print("Constraint %d, type %s, %s is: %f" % (i, conds[i]['type'], conds[i]['info'], conds[i]['fun'](x)))

def constraints_for_links(ng, mode=0, argA=None, argB=None):
    """ mode 0, A and B are all variables
        mode 1, B is variable, A is argA
        mode 2, A is variable, B is argB. """
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    constraints = []
    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)
    C = np.zeros((l_len, s_len*s_len), float)
    for k in range(l_len):
        for i in range(s_len):
            for j in range(s_len):
                if l_list[k] in ng.getShortestPath(s_list[i], s_list[j]).getLinkList():
                    C[k,i*s_len+j] = 1
    # print("C in constraints_for_links")
    # print(C)
    bandwidth = np.zeros(l_len)
    for k in range(l_len):
        bandwidth[k] = l_list[k].getBandwidth()
    # print("Bandiwidth list for links:\n", bandwidth)
    
    # constraints for bandwidth of links
    if mode == 0:
        for k in range(l_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[s_len**2:2*s_len**2], C[k,:]) - ADJUST
            tmp['info'] = 'remained bandwith of %s ' % (l_list[k].getKey())
            constraints.append(tmp)
            # print("tmp fun:", tmp['fun'](x))
    elif mode == 2:
        constraints = []
    elif mode == 1:
        for k in range(l_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[0:s_len**2], C[k,:]) - ADJUST
            tmp['info'] = 'remained bandwith of %s ' % (l_list[k].getKey())
            constraints.append(tmp)
    elif mode == 3:
        for k in range(l_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[0:s_len**2]*VA_DATA_MEAN, C[k,:]) - ADJUST
            tmp['info'] = 'remained bandwith of %s ' % (l_list[k].getKey())
            constraints.append(tmp)
    else: 
        print("No such mode.")
        sys.exit(3)

    return constraints

def constraints_for_sender(ng, mode=0, argA=None, argB=None):
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    constraints = []
    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)
    # constraints for load of offloading
    if mode == 0 or mode == 2 or mode == 3:
        for i in range(s_len):
            tmp = {}
            tmp['type'] = 'eq'
            tmp['fun'] = lambda x, i=i: -np.sum(x[i*s_len:(i+1)*s_len]) + alpha[i] # + ADJUST
            tmp['info'] = 'Remaining load of %s' % (s_list[i].getKey())
            constraints.append(tmp)
    elif mode == 1:
        constraints = []
    else:
        print("No such mode")
        sys.exit(4)

    return constraints
    # pass

def constraints_for_sender_var(ng, mode=0, argA=None, argB=None):
    """ Just change eq to ineq """
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    constraints = []
    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)
    # constraints for load of offloading
    if mode == 0 or mode==2 or mode == 3:
        for i in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i: - np.sum(x[i*s_len:(i+1)*s_len]) + alpha[i] - ADJUST
            tmp['info'] = 'sended load of %s' % (s_list[i].getKey())
            constraints.append(tmp)
    elif mode == 1:
        constraints = []
    return constraints

def constraints_for_sender_var2(ng, mode=0, argA=None, argB=None):
    """ Just change eq to ineq """
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    constraints = []
    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)
    # constraints for load of offloading
    if mode == 0 or mode==2:
        for i in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i] + 10*ADJUST
            tmp['info'] = 'sended load of %s' % (s_list[i].getKey())
            constraints.append(tmp)
    elif mode == 1:
        constraints = []
    return constraints

def constraints_for_receiver(ng, mode=0, argA=None, argB=None):
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    # print('mu:')
    # print(mu)
    constraints = []
    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)
    H = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            H[j, i*s_len + j] = 1
    if mode == 0:
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,:]) - ADJUST
            tmp['info'] = 'remained rsc of %s' % (s_list[j].getKey())
            constraints.append(tmp)
    elif mode == 1:
        constraints = []
    elif mode == 2 or mode == 3:
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,0:s_len**2]) - ADJUST
            tmp['info'] = 'remained rsc of %s' % (s_list[j].getKey())
            constraints.append(tmp)
    return constraints
    # pass

def constraints_for_path_ban(ng, mode=0, argA=None, argB=None):
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    constraints = []
    # if mode == 1 and argA.all() == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)
    # Constraint for bandwidth of path
    if mode == 0:
        for i in range(s_len):
            for j in range(s_len):
                tmp = {}
                tmp['type'] = 'ineq'
                tmp['fun'] = lambda x, i=i, j=j: x[s_len**2+i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN - ADJUST
                tmp['info'] = 'surplus bandwith of path from %s to %s' %(s_list[i].getKey(), s_list[j].getKey()) 
                constraints.append(tmp)
    elif mode == 1:
        for i in range(s_len):
            for j in range(s_len):
                tmp = {}
                tmp['type'] = 'ineq'
                tmp['fun'] = lambda x, i=i, j=j, argA=argA: x[i*s_len+j] - argA[i*s_len+j]*VA_DATA_MEAN - ADJUST
                tmp['info'] = 'surplus bandwith of path from %s to %s' %(s_list[i].getKey(), s_list[j].getKey())
                constraints.append(tmp)
    elif mode == 2:
        for i in range(s_len):
            for j in range(s_len):
                tmp = {}
                tmp['type'] = 'ineq'
                tmp['fun'] = lambda x, i=i, j=j, argB=argB: argB[i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN # - ADJUST 这里不能减去ADJEST，否则当路径带宽为0时，就要求卸载量是负的，就和卸载量必须为正的条件冲突，引发“Inequality constraints incompatible"错误
                tmp['info'] = 'surplus bandwith of path from %s to %s' %(s_list[i].getKey(), s_list[j].getKey()) 
                constraints.append(tmp)
    elif mode == 3:
        constraints = []
    else:
        print("No such mode")
        sys.exit(4)
    return constraints
    # pass

def constraints_for_offloading(ng, mode=0, argA=None, argB=None):
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    constraints = []
    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)

    # Constraint for offloading decision
    if mode == 0 or mode ==2 or mode == 3:
        for i in range(s_len):
            for j in range(s_len):
                tmp = {}
                tmp['type'] = 'ineq'
                tmp['fun'] = lambda x, i=i, j=j: x[i*s_len+j] - ADJUST/20 # 减去ADJUST或可以提高SLSQP算法效果，并且如果不用，可能导致PVI求出不符合约束的解。
                tmp['info'] = 'offloading from %s to %s' %(s_list[i].getKey(), s_list[j].getKey())
                constraints.append(tmp)
    elif mode ==1 :
        constraints = []
    else:
        # print("No such mode")
        raise ValueError("No such mode")
    return constraints
    # pass

def get_ng_info_for_constraints(ng):
    s_list = ng.getServerList()
    l_list = ng.getLinkList()
    s_len = len(s_list)
    l_len = len(l_list)
    alpha = np.zeros(s_len, float)
    mu = np.zeros(s_len, float)
    # x = np.zeros(2*len(s_list)**2, float)+1.522222
    for i in range(s_len):
        alpha[i] = s_list[i].getGroup().getTaskGenInfo(CODE_TASK_TYPE_VA)[0] # Get the mean of distribution
        mu[i] = s_list[i].getRscAmount()/VA_COM_MEAN
    return s_list, l_list, s_len, l_len, alpha, mu

def gen_intact_constraints(ng):
    
    s_list = ng.getServerList()
    l_list = ng.getLinkList()
    s_len = len(s_list)
    l_len = len(l_list)
    C = np.zeros((l_len, s_len*s_len), float)
    for k in range(l_len):
        # tmp = []
        # not_empty = 0
        for i in range(s_len):
            for j in range(s_len):
                if l_list[k] in ng.getShortestPath(s_list[i], s_list[j]).getLinkList():
                    C[k,i*s_len+j] = 1
    # print("Constraint matrix is:")
    # for k in range(l_len):
    #     print(C[k,:])
    # for k in range(l_len):
    #     print(sum(C[k,:]))

    # Get task generating info 
    alpha = np.zeros(s_len, float)
    mu = np.zeros(s_len, float)
    x = np.zeros(2*len(s_list)**2, float)+1.522222
    for i in range(s_len):
        alpha[i] = s_list[i].getGroup().getTaskGenInfo(CODE_TASK_TYPE_VA)[0] # Get the mean of distribution
        mu[i] = s_list[i].getRscAmount()/VA_COM_MEAN
    print("alpha:",alpha)

    bandwidth = np.zeros(l_len)
    for k in range(l_len):
        bandwidth[k] = l_list[k].getBandwidth()
    print("Bandiwidth list for links:\n", bandwidth)
    constraints = []
    # constraints for bandwidth of links
    """ for k in range(l_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[s_len**2:2*s_len**2], C[k,:])
        tmp['info'] = 'link'
        constraints.append(tmp) """
    tmp = {}
    tmp['type'] = 'ineq'
    tmp['fun'] = lambda x: bandwidth -  np.dot(C, x[s_len**2:2*s_len**2]) - ADJUST
    constraints.append(tmp)
       
    #constraints for load of offloading
    """ for i in range(s_len):
        tmp = {}
        tmp['type'] = 'eq'
        tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i]
        tmp['info'] = 'sender'
        constraints.append(tmp) """
        # print("tmp fun:", tmp['fun'](x))
    M = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            M[i, i*s_len:(i+1)*s_len] = 1
    tmp = {}
    tmp['type'] = 'eq'
    tmp['fun'] = lambda x: np.dot(M, x) - alpha  - ADJUST
    constraints.append(tmp)

    print("mu:", mu)
    # constraints for receivers' load
    H = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            H[j, i*s_len + j] = 1
    """ for j in range(s_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,:])
        tmp['info'] = 'receiver'
        constraints.append(tmp) """
    tmp = {}
    tmp['type'] = 'ineq'
    tmp['fun'] = lambda x: mu -  np.dot(H, x)  - ADJUST
    constraints.append(tmp)

    # Constraint for bandwidth of path
    """ for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[s_len**2+i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN
            tmp['info'] = 'path'
            constraints.append(tmp) """
            # print("tmp fun:", tmp['fun'](x))
    Q = np.zeros((s_len**2, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            Q[i*s_len+j, s_len**2+i*s_len+j] = 1
            Q[i*s_len+j, i*s_len+j] = -VA_DATA_MEAN
    tmp = {}
    tmp['type'] = 'ineq'
    tmp['fun'] = lambda x: np.dot(Q, x)  - ADJUST
    constraints.append(tmp)

    # Constraint for offloading decision
    """ for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[i*s_len+j]
            tmp['info'] = 'offloading'
            constraints.append(tmp) """
    P = np.zeros((s_len**2, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            P[i*s_len+j, i*s_len+j] = 1
    tmp = {}
    tmp['type'] = 'ineq'
    tmp['fun'] = lambda x: np.dot(P, x)
    constraints.append(tmp)
    
    # print("constraints:")
    # for c in constraints:
    #     print(c)

    return constraints

def gen_one_constraint(ng, mode=0):
    """ Merge all constraints into one formular, 
    be careful that eq constraints are changed into ineq's """
    s_list = ng.getServerList()
    l_list = ng.getLinkList()
    s_len = len(s_list)
    l_len = len(l_list)
   
    alpha = np.zeros(s_len, float)
    mu = np.zeros(s_len, float)
    x = np.zeros(2*len(s_list)**2, float)+1.522222
    for i in range(s_len):
        alpha[i] = s_list[i].getGroup().getTaskGenInfo(CODE_TASK_TYPE_VA)[0] # Get the mean of distribution
        mu[i] = s_list[i].getRscAmount()/VA_COM_MEAN

    bandwidth = np.zeros(l_len)
    for k in range(l_len):
        bandwidth[k] = l_list[k].getBandwidth()
    print("alpha:\n",alpha)
    print("Bandiwidth list for links:\n", bandwidth)
    print("mu:\n", mu)

    # constraints for bandwidth of links
    C = np.zeros((l_len, 2*s_len*s_len), float)
    for k in range(l_len):
        # tmp = []
        # not_empty = 0
        for i in range(s_len):
            for j in range(s_len):
                if l_list[k] in ng.getShortestPath(s_list[i], s_list[j]).getLinkList():
                    C[k,s_len**2+i*s_len+j] = 1

    #constraints for load of offloading
    M = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            M[i, i*s_len:(i+1)*s_len] = 1

    # constraints for receivers' load
    H = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            H[j, i*s_len + j] = 1

    # Constraint for bandwidth of path
    Q = np.zeros((s_len**2, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            Q[i*s_len+j, s_len**2+i*s_len+j] = 1
            Q[i*s_len+j, i*s_len+j] = -VA_DATA_MEAN

    # Constraint for offloading decision
    P = np.zeros((s_len**2, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            P[i*s_len+j, i*s_len+j] = 1
    
    h_1 = np.zeros(s_len**2, float) -ADJUST
    h_2 = np.zeros(s_len**2)
    A = np.concatenate((-C,M,-H,Q,P), axis=0)
    b = np.concatenate((bandwidth -ADJUST,-alpha -ADJUST, mu -ADJUST, h_1, h_2)) 
    if mode == 1:
        return A, b
    elif mode == 0:
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x: np.dot(A, x) + b

        return [tmp]

def gen_constraints(ng):
    """ Generating constraint one by one """
    s_list = ng.getServerList()
    l_list = ng.getLinkList()
    s_len = len(s_list)
    l_len = len(l_list)
    C = np.zeros((l_len, s_len*s_len), float)
    for k in range(l_len):
        for i in range(s_len):
            for j in range(s_len):
                if l_list[k] in ng.getShortestPath(s_list[i], s_list[j]).getLinkList():
                    C[k,i*s_len+j] = 1


    # Get task generating info 
    alpha = np.zeros(s_len, float)
    mu = np.zeros(s_len, float)
    x = np.zeros(2*len(s_list)**2, float)+1.522222
    for i in range(s_len):
        alpha[i] = s_list[i].getGroup().getTaskGenInfo(CODE_TASK_TYPE_VA)[0] # Get the mean of distribution
        mu[i] = s_list[i].getRscAmount()/VA_COM_MEAN
    print("alpha:",alpha)

    bandwidth = np.zeros(l_len)
    for k in range(l_len):
        bandwidth[k] = l_list[k].getBandwidth()
    print("Bandiwidth list for links:\n", bandwidth)
    constraints = []

    # Constraint for bandwidth of link
    for k in range(l_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[s_len**2:2*s_len**2], C[k,:]) - ADJUST # Adjust ban a little lower to avoid exceeding link bandwidth.
        tmp['info'] = "remaining bandwidth of %s" % l_list[k].getKey()
        constraints.append(tmp)

    # Constraint for sender's load
    for i in range(s_len):
        tmp = {}
        tmp['type'] = 'eq'
        tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i] - ADJUST
        tmp['info'] = "remaning load of %s" % s_list[i].getKey()
        constraints.append(tmp)
    
    # Constraint for receiver's load
    H = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            H[j, i*s_len + j] = 1
    for j in range(s_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,:]) - ADJUST # Adjust load a little lower to avoid exceeding server rsc
        tmp['info'] = "remaining rsc of %s" % s_list[j].getKey()
        constraints.append(tmp)

    # Constraint for bandwidth of path
    for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[s_len**2+i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN - ADJUST # Adjust ban a little higher to avoid zero bandwidth.
            tmp['info'] = "surplus bandwidth of path from %s to %s" % (s_list[i].getKey(), s_list[j].getKey())
            constraints.append(tmp)

    # Constraint for offloading decision
    for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[i*s_len+j]# - ADJUST/100
            tmp['info'] = "offloading from %s to %s" % (s_list[i].getKey(), s_list[j].getKey())
            constraints.append(tmp)

    return constraints

    # return constraints
    # print(constraints)


# Generating feasible solution
def gen_feasible_solution_exterior(cons, args, cons_mode=0, true_initial=False):
    print("generating feasible solution by exterior_penalty:")
    if cons_mode == 0: # If each element in cons is just one constraint, not a set of constraints.
        def penalty_func(x):
            f = 0
            for e in cons:
                if e['type'] == 'eq':
                    # pass
                    f = e['fun'](x)**2*PENALTY + f
                elif e['type'] == 'ineq':
                    f = max(0, -e['fun'](x))**2*PENALTY + f
                    # f = 1/e['fun'](x)*0.001 + f
            # func = gen_objective(args)
            return f # + func(x)
    elif cons_mode == 1: # If all constraints are merged into one
        def penalty_func(x):
            vec = cons[0]['fun'](x)
            f = 0
            for s in vec:
                if s < 0:
                    f = f + s**2*PENALTY
            return f
    elif cons_mode == 2: # if all constraints are divided into five class.
        def penalty_func(x):
            f = 0
            for con in cons: 
                vec = con['fun'](x)
                if con['type'] == 'eq':
                    for s in vec:
                        f = f + s**2*PENALTY
                elif con['type'] == 'ineq':
                    for s in vec:
                        if s < 0:
                            f = f + s**2*PENALTY
            return f
    else:
        print("Wrong cons_mode")
        sys.exit(6)
    
    server_num, sigma_d, beta_d, sigma_h, beta_h, c, path_bans = args
    
    if true_initial == True:
        for i  in range(400):
            print("count: ", i)
            x0 = np.random.rand(2*server_num**2)
            res = minimize(penalty_func, x0, method='SLSQP',constraints=cons, tol=0.0001)
            if res.success ==True:
                break
        if i==399:
            # print("Iteration for finding initial value exceeds 200 times.")
            raise ValueError("Iteration for finding initial value exceeds 400 times.")
    else:
        x0 = np.random.rand(2*server_num**2)
        res = minimize(penalty_func, x0, method='SLSQP',constraints=cons, tol=0.0001)
    # print(res)
    # print(res.fun)
    # print(res.x)
    # print(res.message)
    # check_constraints(cons, res.x)
    print(res.success)

    return res

def gen_feasible_solution_interior(cons, args):
    print("Generating feasible solution by interior_penalty:")
    in_func = interior_penalty(cons)
    server_num, sigma_d, beta_d, sigma_h, beta_h, c, path_bans = args
    # flag = True
    # num = 1
    # while flag:
    #     print("--------Iteration %d" % (num))

    x0 = np.random.rand(2*server_num**2)

    res = minimize(in_func, x0, method='SLSQP',constraints=cons, tol=0.0001)
    # print(res)
    # check_constraints(cons, res.x)
    print(res.success)

    return res

def gen_feasible_solution_linear(cons, args, cons_mode=0, true_initial=False, mode=0):
    server_num, sigma_d, beta_d, sigma_h, beta_h, c, path_bans = args
    if true_initial == True:
        for i  in range(100):
            print("count: ", i)
            if mode == 0:
                x0 = np.random.rand(2*server_num**2)
            elif mode == 2 or mode == 1:
                x0 = np.random.rand(server_num**2)
            res = minimize(linear_obj, x0, method='SLSQP',constraints=cons, tol=0.0001)
            if res.success ==True:
                break
        if i==100:
            print("Iteration for finding initial value exceeds 100 times.")
            sys.exit(10)
    else:
        x0 = np.random.rand(2*server_num**2)
        res = minimize(linear_obj, x0, method='SLSQP',constraints=cons, tol=0.0001)
    
    print(res.success)

    return res

def gen_feasible_solution_com(cons, args, cons_mode=0, true_initial=False):
    server_num, sigma_d, beta_d, sigma_h, beta_h, c, path_bans = args
    obj = gen_objective(args, mode=4)
    if true_initial == True:
        for i  in range(100):
            print("count: ", i)
            x0 = np.random.rand(2*server_num**2)*5
            res = minimize(obj, x0, method='SLSQP',constraints=cons, tol=0.0001)
            if res.success ==True:
                break
        if i==99:
            print("Iteration for finding initial value exceeds 100 times.")
            sys.exit(10)
    else:
        x0 = np.random.rand(2*server_num**2)
        res = minimize(linear_obj, x0, method='SLSQP',constraints=cons, tol=0.0001)
    
    print(res.success)

    return res

def gen_feasible_solution_zero(cons, args, cons_mode=0, true_initial=False):
    server_num, sigma_d, beta_d, sigma_h, beta_h, c, path_bans = args
    def obj(x):
        return 0
    if true_initial == True:
        for i  in range(100):
            print("count: ", i)
            x0 = np.random.rand(2*server_num**2)
            res = minimize(obj, x0, method='SLSQP',constraints=cons, tol=0.0001)
            if res.success ==True:
                break
        if i==100:
            print("Iteration for finding initial value exceeds 100 times.")
            sys.exit(10)
    else:
        x0 = np.random.rand(2*server_num**2)
        res = minimize(linear_obj, x0, method='SLSQP',constraints=cons, tol=0.0001)
    
    print(res.success)

    return res

def interior_penalty(cons):
    def func(x):
        f = 0
        for e in cons:
            if e['type'] == 'ineq':
                f = f + 1/e['fun'](x)
            else:
                print("interior can't handle equation constraint.")
                sys.exit(1)
        return f
    return func

def exterior_penalty(cons, penalty):
    def func(x):
        f = 0
        for e in cons:
            if e['type'] == 'eq':
                f = f + e['fun'](x)**2*penalty
            elif e['type'] == 'ineq':
                f = f + max(0, -e['fun'](x))**2*penalty
            # else:
            #     print("Unknow type of constraint, type name is %s" % (e['type']))
            #     sys.exit(2)
        # if PENALTY > 1000:
        #     PENALTY = PENALTY * DISCOUNT
        return f
    return func

def linear_obj(x):
    return np.sum(x)

def rectify(x):
    for i in range(len(x)):
        if x[i] <0:
            x[i] = 0
    return x


# Some small tools
def reshape_x(x, server_num):
    A = np.zeros((server_num,server_num))
    B = np.zeros((server_num,server_num))

    for i in range(server_num):
        for j in range(server_num):
            A[i,j] = x[i*server_num + j]
            B[i,j] = x[server_num**2 + i*server_num + j]
    return A, B

def gen_constraints_from_five_class(tng):
    constraints = constraints_for_links(tng)
    constraints = constraints_for_sender(tng) + constraints
    constraints = constraints_for_receiver(tng) + constraints
    constraints = constraints_for_path_ban(tng) + constraints
    constraints = constraints_for_offloading(tng) + constraints

    return constraints

def print_result(x, s_len):
    A, B = reshape_x(x, s_len)
    print("Offloading decisions:\n", A)
    print("Receivers' load:\n", np.sum(A,axis=0))
    print("Sender's load:\n", np.sum(A, axis=1))
    print("Path bandwith allocation:\n", B)

# Test functions
def test_gen_feasible_zero(tng, max_iteration=200, repeat=5):
    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)
    # partial_iteration = 50
    # repeat = 5
    f = gen_objective(args)
    total = 0
    for i in range(repeat):
        result_for_slsqp = []
        x0 = gen_feasible_solution_zero(constraints1, args, cons_mode=0, true_initial=True).x
        # x0 = np.random.rand(2*s_len**2)
        # res = minimize(gen_objective(args), x0, method='SLSQP',constraints=constraints1, callback=callback)
        # print(res)
        print(f(x0))
        check_constraints(constraints1, x0)
        total = total + f(x0)
    print("Average feasible: ",total/repeat)

def test_gen_feasible_linear(tng, max_iteration=200, repeat=5):
    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)
    # partial_iteration = 50
    # repeat = 5
    f = gen_objective(args)
    total = 0
    for i in range(repeat):
        result_for_slsqp = []
        x0 = gen_feasible_solution_linear(constraints1, args, cons_mode=0, true_initial=True).x
        # x0 = np.random.rand(2*s_len**2)
        # res = minimize(gen_objective(args), x0, method='SLSQP',constraints=constraints1, callback=callback)
        # print(res)
        print(f(x0))
        check_constraints(constraints1, x0)
        total = total + f(x0)
    print("Average feasible: ",total/repeat)

def test_gen_feasible_penalty(tng, max_iteration=200, repeat=5):
    
    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)
    # partial_iteration = 50
    # repeat = 5
    f = gen_objective(args)
    total = 0
    # def callback(xk):
    #     result_for_slsqp.append(xk)
    for i in range(repeat):
        result_for_slsqp = []
        x0 = gen_feasible_solution_exterior(constraints1, args, cons_mode=0, true_initial=True).x
        # x0 = np.random.rand(2*s_len**2)
        # res = minimize(gen_objective(args), x0, method='SLSQP',constraints=constraints1, callback=callback)
        # print(res)
        print(f(x0))
        total = total + f(x0)
    print("Average feasible: ",total/repeat)

def observe_feasible_solution_feature(tng, max_iteration=200, repeat=5):
    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)

    gen_feasible_func = gen_feasible_solution_linear
    # partial_iteration = 50
    # repeat = 5
    f = gen_objective(args)

    objective_for_slsqp = np.zeros(max_iteration)
    success = 0
    # 用来记录最终结果
    results = np.zeros(repeat)
    s_list = tng.getServerList()
    s_len = len(s_list)
    def callback(xk):
        result_for_slsqp.append(xk)

    obj = gen_objective(args)
    while success < repeat:
        result_for_slsqp = []
        x0 = gen_feasible_func(constraints1, args, cons_mode=0, true_initial=True).x
        # check_constraints(constraints1, x0)
        # x0 = rectify(x0)
        res = minimize(obj, x0, method='SLSQP',constraints=constraints1, callback=callback)
        # print(res)
        if res.success ==True and res.fun >=0 and res.fun < 1000: # and res.fun < 1:
            
            print("-------------constraints satisfaction of initial x-----------")
            check_constraints(constraints1, x0)
            print("obj of initial result: ",obj(x0))
            results[success] = res.fun

            success = success + 1
            print("success count: ", success)
            print(res)
            print_result(res.x, len(s_list))
            print("difference between initial and final:")
            print_result(res.x-x0, len(s_list))
            print("double check:")
            res = minimize(obj, x0, method='SLSQP',constraints=constraints1, callback=callback)
            print(res)

        else:
            print("Bad repeat")




def compare_numerical_method_performance(tng, max_iteration=200, repeat=5, seed=1, file_prefix='', need_timing=False, gen_feasible='penalty'):
    np.random.seed(seed)
    # max_iteration = 200
    # repeat = 20
    tc = {}
    results_for_lower =  test_lower_bound(tng, repeat=repeat, max_iteration=max_iteration, gen_feasible=gen_feasible)
    np.savetxt('%slower_result.txt' % file_prefix, results_for_lower, fmt = '%f')
    results_for_partial, tc['pvi'] = test_partial_iteration(tng, repeat=repeat, max_iteration=max_iteration, return_result=True, need_timing=need_timing, gen_feasible=gen_feasible)
    np.savetxt('%spartial_result.txt' % file_prefix, results_for_partial, fmt = '%f')
    results_for_normal, tc['slsqp'] = test_normal_iteration(tng, max_iteration=max_iteration, repeat=repeat, return_result=True, need_timing=need_timing, gen_feasible=gen_feasible)
    np.savetxt('%snormal_result.txt' % file_prefix, results_for_normal, fmt = '%f')
    results_for_seperate = test_partial_iteration_with_max_ban(tng, max_iteration=max_iteration, repeat=repeat, return_result=True, gen_feasible=gen_feasible)
    np.savetxt('%sseperate_result.txt' % file_prefix, results_for_seperate, fmt = '%f')
    if need_timing == True:
        return tc

def test_lower_bound(tng, max_iteration=200, repeat=5, gen_feasible='penalty'):
    # 首先生成约束
    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    # 确定求解可行解的方法
    args = gen_args(tng)
    if gen_feasible == 'penalty':
        gen_feasible_func = gen_feasible_solution_exterior
    elif gen_feasible == 'linear':
        gen_feasible_func = gen_feasible_solution_linear
    elif gen_feasible == 'com':
        gen_feasible_func = gen_feasible_solution_com
    else:
        raise ValueError("No generating feasible solution function corresponding to %s " % gen_feasible)

    # 准备一些局部变量
    success = 0
    results = np.zeros(repeat)

    # 设置回调函数用来记录迭代历史
    def callback(xk):
        result_for_slsqp.append(xk)
    while success < repeat:
        result_for_slsqp = []

        # 首先生成可行解
        x0 = gen_feasible_func(constraints1, args, true_initial=True).x

        # 目标函数取mode=4，也就是只考虑计算时间
        resA = minimize(gen_objective(args, mode=4), x0, method='SLSQP',constraints=constraints1, callback=callback)#, options={ 'maxiter':300})# , options={"maxiter":1})
        
        if resA.success != True or resA.fun > 1000:
            print("At A, Bad repeat, objective is %f" % resA.fun)
            continue

        # 目标函数取5
        resB = minimize(gen_objective(args, mode=5), x0, method='SLSQP',constraints=constraints1, options={'maxiter':max_iteration}, callback=callback)# options={"maxiter":1})
        # print(resB)
        if (resB.success == False and resB.message != 'Iteration limit exceeded') or resB.fun > 1000:
            print("At B, Bad repeat, objective is %f" % resB.fun)
            continue

        # objective_for_pvi = objective_for_pvi + record_fun
        results[success] = resA.fun + resB.fun
        success = success + 1
        print("success count %d" % success)
        print('resA:\n', resA)
        print("resB:\n", resB)

        
        
       
    return results

def test_normal_iteration(tng, max_iteration, repeat=5, gen_feasible='penalty', return_result=False, need_timing=False):
    # np.random.seed(9)


    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)

    if gen_feasible == 'penalty':
        gen_feasible_func = gen_feasible_solution_exterior
    elif gen_feasible == 'linear':
        gen_feasible_func = gen_feasible_solution_linear
    elif gen_feasible == 'com':
        gen_feasible_func = gen_feasible_solution_com
    else:
        raise ValueError("No generating feasible solution function corresponding to %s " % gen_feasible)
    
    time_consumption = []

    f = gen_objective(args)

    objective_for_slsqp = np.zeros(max_iteration)
    success = 0
    # 用来记录最终结果
    results = np.zeros(repeat)
    s_list = tng.getServerList()
    s_len = len(s_list)
    def callback(xk):
        result_for_slsqp.append(xk)

    start_time = time.time()
    while success < repeat:
        result_for_slsqp = []
        x0 = gen_feasible_func(constraints1, args, cons_mode=0, true_initial=True).x
        # x0 = np.random.rand(2*s_len**2)
        res = minimize(gen_objective(args), x0, method='SLSQP',constraints=constraints1, callback=callback, options={"maxiter":300, 'eps':0.005})#, jac=gen_deriv_for_obj(args=args, mode=0))
        # print(res)
        if res.success ==True and res.fun >=0 and res.fun < 1000: # and res.fun < 1:

            # 记录时间消耗
            end_time = time.time()
            time_consumption.append(end_time - start_time)
            start_time = end_time

            # 记录收敛结果
            results[success] = res.fun

            success = success + 1
            print("success count: ", success)

            print(res)
            print_result(res.x, len(s_list))

            for i in range(len(objective_for_slsqp)):
                if i >= len(result_for_slsqp):
                    objective_for_slsqp[i] = objective_for_slsqp[i] + f(res.x)
                else:
                    objective_for_slsqp[i] = f(result_for_slsqp[i]) + objective_for_slsqp[i]
        else:
            print("Bad repeat")
                
    objective_for_slsqp = objective_for_slsqp/success

    if return_result == True:
        if need_timing == False:
            return results
        else:
            return results, time_consumption
    else:
        return objective_for_slsqp

def test_partial_iteration(tng, max_iteration, repeat=5, gen_feasible='penalty', return_result=False, need_timing=False):
    # partial variable iteration doesn't work well, or  pretty bad.
    # np.random.seed(9)
    print("In test_partial_iteration:")

    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)

    if gen_feasible == 'penalty':
        gen_feasible_func = gen_feasible_solution_exterior
    elif gen_feasible == 'linear':
        gen_feasible_func = gen_feasible_solution_linear
    elif gen_feasible == 'com':
        gen_feasible_func = gen_feasible_solution_com
    else:
        raise ValueError("No generating feasible solution function corresponding to %s " % gen_feasible)
    # partial_iteration = 50
    normal_iteration = max_iteration
    f = gen_objective(args)

    results = np.zeros(repeat)

    time_consumption = []
    
    s_list = tng.getServerList()
    objective_for_pvi = np.zeros(max_iteration)
    success = 0
    # If we use partial variable iteration
    start_time = time.time()

    old_fun = NUM_FLOAT_INFINITY
    while success < repeat:
        flag = 0
        record_fun = []

        # Generate initial value
        x0 = gen_feasible_func(constraints1, args, true_initial=True).x
        # x0 = np.random.rand(2*len(s_list)**2)
        argA = x0[0:len(s_list)**2]
        argB = x0[len(s_list)**2: 2*len(s_list)**2]
        
        for i in range(max_iteration):
            if i % 2 == 0:
                cons = []
                cons = constraints_for_links(tng, mode=2, argB=argB)
                cons = constraints_for_offloading(tng, mode=2, argB=argB) + cons
                cons = constraints_for_path_ban(tng, mode=2, argB=argB) + cons
                cons = constraints_for_receiver(tng, mode=2, argB=argB) + cons
                cons = constraints_for_sender(tng, mode=2, argB=argB) + cons
                # argA = gen_feasible_solution_linear(cons, args=args, true_initial=True, mode=2).x
                resA = minimize(gen_objective(args, mode=2, argB=argB), argA, method='SLSQP',constraints=cons, options={"maxiter":3})#, jac=gen_deriv_for_obj(args, mode=2, argB=argB))# , options={"maxiter":1})
                # print("Offloading :")
                # print((resA.fun))
                # print(resA.success)
                # print(resA.message)
                # if resA.fun <0 or resA.fun>1000 or (resA.success != True and resA.message != "Iteration limit exceeded"):
                if resA.fun <0 or (resA.success != True and resA.message != "Iteration limit exceeded"):
                    flag = 1
                    print("Iteration %d, bad repeat in mode 2, obj %f, %s" % (i,resA.fun, resA.message))
                    # tmpx = np.zeros(len(resA.x))
                    # check_constraints(cons, resA.x)
                    break
                argA = resA.x
                record_fun.append(resA.fun)
                # record_f.append(f(np.concatenate((np.array(argA),np.array(argB)))))
            # check_constraints(cons, resA.x)
            # print(resA)
            else:
                cons = []
                cons = constraints_for_links(tng, mode=1, argA=argA)
                cons = constraints_for_offloading(tng, mode=1, argA=argA) + cons
                cons = constraints_for_path_ban(tng, mode=1, argA=argA) + cons
                cons = constraints_for_receiver(tng, mode=1, argA=argA) + cons
                cons = constraints_for_sender(tng, mode=1, argA=argA) + cons
                # resB = minimize(gen_objective(args, mode=1, argA=argA), argB, method='SLSQP',constraints=cons, options={"maxiter":1})# options={"maxiter":1})
                # argB = gen_feasible_solution_linear(cons, args=args, true_initial=True, mode=1).x
                resB = minimize(gen_objective(args, mode=1, argA=argA), argB, method='SLSQP',constraints=cons, options={"maxiter":3})#, jac=gen_deriv_for_obj(args, mode=1, argA=argA))
                # print("Bandwidth :")
                # print((resB.fun))
                # print(resB.success)
                if resB.fun <0 or resB.fun >1000 or (resB.success != True and resB.message != "Iteration limit exceeded"):
                    flag = 1
                    # print("Bad repeat, %s" % resB.message)
                    print("Iteration %d, bad repeat in mode 1, obj %f, %s" % (i, resB.fun, resB.message))
                    # tmpx = np.zeros(len(resB.x))
                    # check_constraints(cons, resB.x)
                    break
                argB = resB.x
                record_fun.append(resB.fun)
                
                # 检查是否已经收敛
                # if abs(resB.fun - old_fun) < 0.001:
                #     break
                # old_fun = resB.fun

                
        if flag == 0 and resB.fun < 1000:
            objective_for_pvi = objective_for_pvi + record_fun
            results[success] = resB.fun
            success = success + 1

            # 统计时间
            end_time = time.time()
            time_consumption.append(end_time - start_time)
            start_time = end_time

            # 输出收敛细节
            print("res:")
            print(resB)
            # check_constraints(cons, resB.x)
            print("Repeation ", success)
            print("Offloading:")
            print(np.reshape(argA,(len(s_list), len(s_list))))
            print("lambda:\n",np.sum(np.reshape(argA,(len(s_list), len(s_list))),axis=0))
            print("Bandwidth:")
            print(np.reshape(argB,(len(s_list), len(s_list))))
            
    if return_result == True:
        if need_timing == False:
            return results
        else:
            return results, time_consumption
    else:
        return objective_for_pvi/repeat

def test_partial_iteration_with_min_ban(tng, max_iteration, repeat=5, gen_feasible='penalty', return_result=False):
    # np.random.seed(9)

    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)
    if gen_feasible == 'penalty':
        gen_feasible_func = gen_feasible_solution_exterior
    elif gen_feasible == 'linear':
        gen_feasible_func = gen_feasible_solution_linear
    elif gen_feasible == 'com':
        gen_feasible_func = gen_feasible_solution_com
    else:
        raise ValueError("No generating feasible solution function corresponding to %s " % gen_feasible)
    normal_iteration = max_iteration
    f = gen_objective(args)

    results = np.zeros(repeat)
    
    
    objective_for_slsqp = np.zeros(max_iteration)
    success = 0
    s_list = tng.getServerList()
    # If we use partial variable iteration
    def callback(xk):
        result_for_slsqp.append(xk)
    while success < repeat:
        result_for_slsqp = []

        # 首先生成可行解
        x0 = gen_feasible_func(constraints1, args, true_initial=True).x

        # 然后以最小带宽分配，优化卸载决策
        argA = x0[0:len(s_list)**2]
        argB = x0[len(s_list)**2: 2*len(s_list)**2]
        cons = []
        cons = constraints_for_links(tng, mode=3, argB=argB)
        cons = constraints_for_offloading(tng, mode=3, argB=argB) + cons
        cons = constraints_for_path_ban(tng, mode=3, argB=argB) + cons
        cons = constraints_for_receiver(tng, mode=3, argB=argB) + cons
        cons = constraints_for_sender(tng, mode=3, argB=argB) + cons

        resA = minimize(gen_objective(args, mode=3, argB=argB), argA, method='SLSQP',constraints=cons, callback=callback)#, options={ 'maxiter':300})# , options={"maxiter":1})
        
        if resA.success != True:
            print("At A, Bad repeat")
            continue
        # print(resA)
        A_length = len(result_for_slsqp)
        f = gen_objective(args, mode=3, argB=argB)
        for i in range(A_length):
            objective_for_slsqp[i] = f(result_for_slsqp[i]) + objective_for_slsqp[i]
        # 接着优化带宽分配
        argA = resA.x
             
        cons = []
        cons = constraints_for_links(tng, mode=1, argA=argA)
        cons = constraints_for_offloading(tng, mode=1, argA=argA) + cons
        cons = constraints_for_path_ban(tng, mode=1, argA=argA) + cons
        cons = constraints_for_receiver(tng, mode=1, argA=argA) + cons
        cons = constraints_for_sender(tng, mode=1, argA=argA) + cons
        resB = minimize(gen_objective(args, mode=1, argA=argA), argB, method='SLSQP',constraints=cons, options={'maxiter':300}, callback=callback)# options={"maxiter":1})
        print(resB)
        if (resB.success == False and resB.message != 'Iteration limit exceeded') or resB.fun > resA.fun or resB.fun > 1000:
            print("At B, Bad repeat")
            continue
        argB = resB.x

        # objective_for_pvi = objective_for_pvi + record_fun
        results[success] = resB.fun
        success = success + 1
        print('resA:\n', resA)
        print("resB:\n", resB)
        print("Repeation ", success)
        print("Offloading:")
        print(np.reshape(argA,(len(s_list), len(s_list))))
        print("lambda:\n",np.sum(np.reshape(argA,(len(s_list), len(s_list))),axis=0))
        print("Bandwidth:")
        print(np.reshape(argB,(len(s_list), len(s_list))))
        # check_constraints(constraints1, np.concatenate((argA, argB)))
        f = gen_objective(args, mode=1, argA=argA)
        for i in range(len(objective_for_slsqp)-A_length):
                if A_length + i >= len(result_for_slsqp):
                    objective_for_slsqp[A_length + i] = objective_for_slsqp[A_length + i] + f(resB.x)
                else:
                    objective_for_slsqp[A_length + i] = f(result_for_slsqp[A_length + i]) + objective_for_slsqp[A_length + i]
                
    objective_for_slsqp = objective_for_slsqp/success
    # print(objective_for_slsqp)
    if return_result == True:
        return results
    else:
        return objective_for_slsqp

def test_partial_iteration_with_max_ban(tng, max_iteration, repeat=5, gen_feasible='penalty', return_result=False):
    # np.random.seed(9)

    constraints1 = []
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_offloading(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_sender(tng) + constraints1

    args = gen_args(tng)
    if gen_feasible == 'penalty':
        gen_feasible_func = gen_feasible_solution_exterior
    elif gen_feasible == 'linear':
        gen_feasible_func = gen_feasible_solution_linear
    elif gen_feasible == 'com':
        gen_feasible_func = gen_feasible_solution_com
    else:
        raise ValueError("No generating feasible solution function corresponding to %s " % gen_feasible)
    normal_iteration = max_iteration
    f = gen_objective(args)

    results = np.zeros(repeat)
    
    
    objective_for_slsqp = np.zeros(max_iteration)
    success = 0
    s_list = tng.getServerList()
    # If we use partial variable iteration
    def callback(xk):
        result_for_slsqp.append(xk)
    while success < repeat:
        result_for_slsqp = []

        # 首先生成可行解
        x0 = gen_feasible_func(constraints1, args, true_initial=True).x

        # 然后以最小带宽分配，优化卸载决策
        argA = x0[0:len(s_list)**2]
        argB = x0[len(s_list)**2: 2*len(s_list)**2]
        cons = []
        cons = constraints_for_links(tng, mode=3, argB=argB)
        cons = constraints_for_offloading(tng, mode=3, argB=argB) + cons
        cons = constraints_for_path_ban(tng, mode=3, argB=argB) + cons
        cons = constraints_for_receiver(tng, mode=3, argB=argB) + cons
        cons = constraints_for_sender(tng, mode=3, argB=argB) + cons

        resA = minimize(gen_objective(args, mode=6, argB=argB), argA, method='SLSQP',constraints=cons, callback=callback)#, options={ 'maxiter':300})# , options={"maxiter":1})
        
        if resA.success != True:
            print("At A, Bad repeat")
            continue
        # print(resA)
        A_length = len(result_for_slsqp)
        f = gen_objective(args, mode=6, argB=argB)
        for i in range(A_length):
            objective_for_slsqp[i] = f(result_for_slsqp[i]) + objective_for_slsqp[i]
        # 接着优化带宽分配
        argA = resA.x
             
        cons = []
        cons = constraints_for_links(tng, mode=1, argA=argA)
        cons = constraints_for_offloading(tng, mode=1, argA=argA) + cons
        cons = constraints_for_path_ban(tng, mode=1, argA=argA) + cons
        cons = constraints_for_receiver(tng, mode=1, argA=argA) + cons
        cons = constraints_for_sender(tng, mode=1, argA=argA) + cons
        resB = minimize(gen_objective(args, mode=1, argA=argA), argB, method='SLSQP',constraints=cons, options={'maxiter':300}, callback=callback)# options={"maxiter":1})
        # print(resB)
        if (resB.success == False and resB.message != 'Iteration limit exceeded') or resB.fun > 1000:
            print("At B, Bad repeat")
            continue
        argB = resB.x

        # objective_for_pvi = objective_for_pvi + record_fun
        results[success] = resB.fun
        success = success + 1
        print('resA:\n', resA)
        print("resB:\n", resB)
        print("Repeation ", success)
        print("Offloading:")
        print(np.reshape(argA,(len(s_list), len(s_list))))
        print("lambda:\n",np.sum(np.reshape(argA,(len(s_list), len(s_list))),axis=0))
        print("Bandwidth:")
        print(np.reshape(argB,(len(s_list), len(s_list))))
        # check_constraints(constraints1, np.concatenate((argA, argB)))
        f = gen_objective(args, mode=1, argA=argA)
        for i in range(len(objective_for_slsqp)-A_length):
                if A_length + i >= len(result_for_slsqp):
                    objective_for_slsqp[A_length + i] = objective_for_slsqp[A_length + i] + f(resB.x)
                else:
                    objective_for_slsqp[A_length + i] = f(result_for_slsqp[A_length + i]) + objective_for_slsqp[A_length + i]
                
    objective_for_slsqp = objective_for_slsqp/success
    # print(objective_for_slsqp)
    if return_result == True:
        return results
    else:
        return objective_for_slsqp

def test_objective_function(tng):
    cons = []
    cons = constraints_for_links(tng)
    cons = constraints_for_offloading(tng) + cons
    cons = constraints_for_path_ban(tng) + cons
    cons = constraints_for_receiver(tng) + cons
    cons = constraints_for_sender(tng) + cons
    
    obj = gen_objective(gen_args(tng))
    for i in range(100):
        res = gen_feasible_solution_exterior(cons, gen_args(tng))
        print("Is x in field? ",res.success)
        print("objective function value is: ", obj(res.x))
        if res.success == True and obj(res.x) <0:
            print(res)
            check_constraints(cons, res.x)
    
def test_constraints(tng):
    # They are equal in result.
    cons = gen_intact_constraints(tng)
    x = np.ones(2*len(s_list)**2, float)
    print("gen_intact_constraints")
    for c in cons:
        print(c['fun'](x))
    cons = gen_constraints(tng)
    print("gen_constraints")
    for c in cons:
        print(c['fun'](x))
    cons = []
    cons = constraints_for_links(tng)
    cons = constraints_for_offloading(tng) + cons
    cons = constraints_for_path_ban(tng) + cons
    cons = constraints_for_receiver(tng) + cons
    cons = constraints_for_sender(tng) + cons
    print("gen constraint one by one ")
    for c in cons:
        print(c['fun'](x))
    cons = gen_one_constraint(tng)
    print("gen_one_constraint")
    print(cons[0]['fun'](x))

def test_efficiency_of_penalty_funcions(ng):
    # np.random.seed(np.random.randint(0,100,40))
    iteration = 100
    success_ex = 0
    success_in = 0

    constraints2 = constraints_for_links(tng)
    constraints2 = constraints_for_sender_var(tng) + constraints2
    constraints2 = constraints_for_receiver(tng) + constraints2
    constraints2 = constraints_for_path_ban(tng) + constraints2
    constraints2 = constraints_for_offloading(tng) + constraints2

    print("Test exterior:")
    # np.random.seed(5)
    for i in range(iteration):
        res = gen_feasible_solution_exterior(constraints2, gen_args(ng))
        if res.success:
            success_ex = success_ex + 1
    print("Exterior penalty, iteration %d, success %d, ratio %f" % (iteration, success_ex, success_ex/iteration))

    print("Test interior:")
    # np.random.seed(5)
    for i in range(iteration):
        res = gen_feasible_solution_interior(constraints2, gen_args(ng))
        if res.success:
            success_in = success_in + 1
    print("Interior penalty, iteration %d, success %d, ratio %f" % (iteration, success_in, success_in/iteration))

def test_constraints_division_whether_equal(tng):
    """ If we divide constraints into five, are they equal in function? """
    constraints = gen_constraints(tng)

    constraints2 = constraints_for_links(tng)
    constraints2 = constraints_for_sender(tng) + constraints2
    constraints2 = constraints_for_receiver(tng) + constraints2
    constraints2 = constraints_for_path_ban(tng) + constraints2
    constraints2 = constraints_for_offloading(tng) + constraints2

    x = np.ones(2*len(s_list)**2, float)
    # for i in range(len(constraints)):
    #     print("Constraint is:", constraints[i])
    #     print("Constraint %d's value is: %f" % (i,constraints[i]['fun'](x)))
    
    # f = gen_objective(gen_args(tng))
    # print("Objective function:", f(x))
    print("for constraints one:")
    for c in constraints:
        print(c['fun'](x))
    print("for constraints two:")
    for c in constraints2:
        print(c['fun'](x))

def test_penalty_optimization(tng, max_iteration, repeat=5, gen_feasible='penalty'):
    cons = gen_constraints(tng)
    penalty_func = exterior_penalty(cons, PENALTY)

    args = gen_args(tng)
    obj = gen_objective(args)
    success = 0
    obj_for_po = np.zeros(max_iteration)

    def penalized_obj(x):
        return obj(x) + penalty_func(x)

    def callback(xk):
        result_for_po.append(xk)

    ite=1
    while success < repeat and ite <1000:
    # for i in range(20):
        result_for_po = []
        # x0 = np.random.rand((4**2*2))

        x0 = gen_feasible_solution_exterior(cons, args, true_initial=True).x
        res = minimize(penalized_obj, x0, method='SLSQP', callback=callback, options={'maxiter':200})

        print("Penalty:",PENALTY)

        count = 0
        tmp_penalty = PENALTY/2
        # print("penalty_func(x):",penalty_func(res.x))
        # print("obj(x):",obj(res.x))
        penalty_func_value = 1
        flag = 0
        while True:
            result_for_po = []
            # if obj(res.x)
            if count > 4 or tmp_penalty < 1000:
                break
            penalty_func = exterior_penalty(cons, tmp_penalty)
            count = count + 1
            def penalized_obj(x):
                return obj(x) + penalty_func(x)
            print("Penalty:", tmp_penalty)
            # if obj(res.x) > 0 and penalty_func(res.x) < 0.01:
            #     x0 = res.x
            # else:
            #     x0 = gen_feasible_solution_exterior(cons, args, true_initial=True).x
            # x0 = res.x
            x0 = gen_feasible_solution_exterior(cons, args, true_initial=True).x
            # x0 = np.random.rand(len(x0))
            res = minimize(penalized_obj, x0, method='SLSQP', callback=callback, options={'maxiter':200})
            # print(res)
            tmp_penalty = tmp_penalty/2
            print("penalty_func(x):",penalty_func(res.x))
            print("obj(x):",obj(res.x))
            
            if penalty_func(res.x) < 0.5 and obj(res.x) > 0 and obj(res.x) < 3:# and res.success == True:
                flag = 0
                # for e in result_for_po:
                #     if abs(obj(e)) > repeat*2:
                #         flag = 1
                # if flag == 1:
                #     break
                success = success + 1
                print("Success count: ", success)
                print(res)
                print("length of result_for_po:", len(result_for_po))
                check_constraints(cons, res.x)
            
                print_result(res.x, s_len=len(s_list))
                fun = obj(res.x)
                for i in range(max_iteration):
                    if i < len(result_for_po):
                        obj_for_po[i] = obj(result_for_po[i]) + obj_for_po[i]
                    else:
                        obj_for_po[i] = fun + obj_for_po[i]
                break
        ite = ite +1

    obj_for_po = obj_for_po/success
    print("Result is:")
    print(obj_for_po)
    return obj_for_po

def gen_problem_for_R_language(tng):
    args = gen_args(tng)
    server_num, sigma_d, beta_d, sigma_h, beta_h, c = args

    para = [server_num, sigma_d, beta_d, sigma_h, beta_h]
    np.savetxt('./puppy/results/para.txt', para, fmt = '%f')
    np.savetxt('./puppy/results/c.txt', c, fmt = '%f')

    A, lb = gen_one_constraint(tng, mode=1)
    ub = np.ones(len(lb))*1000
    np.savetxt('./puppy/results/A.txt', A, fmt = '%f')
    np.savetxt('./puppy/results/lb.txt', -lb, fmt = '%f')
    np.savetxt('./puppy/results/ub.txt', ub, fmt = '%f')

    xs = np.zeros((INITIAL_VALUE_NUM, 2*len(s_list)**2))
    cons = []
    cons = constraints_for_links(tng)
    cons = constraints_for_offloading(tng) + cons
    cons = constraints_for_path_ban(tng) + cons
    cons = constraints_for_receiver(tng) + cons
    cons = constraints_for_sender(tng) + cons 
    count = 0
    for i in range(INITIAL_VALUE_NUM):
        xs[i,:] = gen_feasible_solution_exterior(cons, args).x
    np.savetxt('./puppy/results/x0.txt', xs, fmt = '%f')

    pa_l = np.zeros(2*server_num**2)
    pa_u = np.ones(2*server_num**2)*1000
    np.savetxt('./puppy/results/pa_l.txt', pa_l, fmt = '%f')
    np.savetxt('./puppy/results/pa_u.txt', pa_u, fmt = '%f')

    mu = np.zeros(server_num)
    omega = np.zeros(server_num)
    for j in range(server_num):
        mu[j] = c[j]/beta_h
        omega[j] = ((sigma_h/c[j])**2*mu[j]**2+1)/(2*mu[j])
    np.savetxt('./puppy/results/mu.txt', mu, fmt = '%f')
    np.savetxt('./puppy/results/omega.txt', omega, fmt = '%f')

def compare_numerical_method_convergence(tng, max_iteration=200, repeat=5, seed=1, file_prefix=''):
    np.random.seed(seed)
    # max_iteration = 200
    # repeat = 20
    obj_for_partial = test_partial_iteration(tng, repeat=repeat, max_iteration=max_iteration)
    obj_for_normal = test_normal_iteration(tng, max_iteration=max_iteration, repeat=repeat)
    obj_for_seperate = test_partial_iteration_with_min_ban(tng, max_iteration=max_iteration, repeat=repeat)
    # obj_for_seperate = test_partial_iteration_with_max_ban(tng, max_iteration=max_iteration, repeat=repeat)

    # obj_for_penalty = test_penalty_optimization(tng, max_iteration=max_iteration,repeat=repeat)
    
    np.savetxt('./puppy/results/convergence/%spartial.txt' % file_prefix, obj_for_partial, fmt = '%f')
    np.savetxt('./puppy/results/convergence/%snormal.txt' % file_prefix, obj_for_normal, fmt = '%f')
    np.savetxt('./puppy/results/convergence/%sseperate.txt' % file_prefix, obj_for_seperate, fmt = '%f')
     # np.savetxt('./puppy/results/%spenalty.txt' % file_prefix, obj_for_penalty, fmt = '%f')

def draw_comparing_result(file_prefix=''):
    obj_for_partial = np.loadtxt('./puppy/results/%spartial.txt' % file_prefix)
    obj_for_normal = np.loadtxt('./puppy/results/%snormal.txt' % file_prefix)
    # obj_for_penalty = np.loadtxt('./puppy/results/%spenalty.txt' % file_prefix)
    obj_for_seperate = np.loadtxt('./puppy/results/%sseperate.txt' % file_prefix)


    plt.title("Comparing between numerical methods") 
    plt.xlabel("Iteration") 
    plt.ylabel("Objective") 
    # length = min([len(obj_for_normal), len(obj_for_partial), len(obj_for_penalty)])
    length = 200
    x = np.arange(1,length+1)
    plt.plot(x, np.log(np.abs(obj_for_partial)+1)[0:length], color="deeppink",linewidth=1,linestyle=':',label='pvi', marker='1', markersize=4)
    plt.plot(x, np.log(np.abs(obj_for_normal)+1)[0:length], color="darkblue",linewidth=1,linestyle='--',label='slsqp', marker='|', markersize=4)
    # plt.plot(x, np.log(np.abs(obj_for_penalty)+1)[len(obj_for_penalty)-length:], color="goldenrod",linewidth=1,linestyle='dashdot',label='pfm', marker='*', markersize=4)
    plt.plot(x, np.log(np.abs(obj_for_seperate)+1)[0:length], color="green",linewidth=1,linestyle='dashdot',label='sep', marker='*', markersize=4)
    plt.plot(x, np.zeros(length), color='tab:pink', linestyle='dotted', linewidth=1)
    plt.legend(loc=1)
    plt.xlim(0,length)
    plt.ylim(2, 5)
    plt.show()




if __name__ == "__main__":
    np.set_printoptions(formatter={'float':'{:.3f}'.format})

    tng = createATreeGraph()
    tng.print_info()

    s_list = tng.getServerList() 
    l_list = tng.getLinkList()
    np.random.seed()

    # observe_feasible_solution_feature(tng)

    # test_gen_feasible_zero(tng)
    # test_gen_feasible_linear(tng)
    # test_gen_feasible_penalty
    #(tng)
    # test_gen_feasible_penalty
    #(tng, repeat=50)
    # test_partial_iteration_with_min_ban(tng, 200, 5, gen_feasible='linear')
    # test_partial_iteration(tng, max_iteration=200, repeat=5, gen_feasible='com')
    # seed = 10
    # file_prefix = '10_servers-seed_' + str(seed)+'-'
    # compare_numerical_method_convergence(tng, max_iteration=200, seed=seed, file_prefix=file_prefix)
    # seed = 1
    # file_prefix = './puppy/results/performance/10_servers-seed_' + str(seed)+'-'
    # compare_numerical_method_performance(tng, max_iteration=200, seed=seed, file_prefix=file_prefix)
    # draw_comparing_result(file_prefix=file_prefix)
    # test_normal_iteration(tng, 200, 5, gen_feasible='penalty')
    # obj_for_penalty  = test_penalty_optimization(tng, 200, repeat=5)
    # np.savetxt('./puppy/results/penalty.txt', obj_for_penalty, fmt = '%f')
    # compare_partial_and_normal(tng)
    # draw_compare_partial_normal()
    # test_constraints(tng)
    # test_objective_function(tng)
    # print(test_partial_iteration(tng, 80))
    # np.savetxt('./puppy/results/partial.txt',test_partial_iteration(tng, 80), fmt = '%f')
    # test_constraints_division_whether_equal(tng)
    # test_efficiency_of_penalty_funcions(tng)

    # gen_problem_for_R_language(tng)

    
    