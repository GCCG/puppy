from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from ..parameters import *
import numpy as np
from scipy.optimize import minimize

# Control task model here
VA_COM_MEAN = 0.1
VA_COM_VAR = 1
VA_DATA_MEAN = 1
VA_DATA_VAR = 1

# Control algorithm parameters here
PENALTY = 1000000


    


def gen_objective(args):
    server_num, sigma_d, beta_d, sigma_h, beta_h, c = args
    mu = np.zeros(server_num)
    omega = np.zeros(server_num)
    for j in range(server_num):
        mu[j] = c[j]/beta_h
        omega[j] = ((sigma_h/c[j])**2*mu[j]**2+1)/(2*mu[j])

    def func(x):
        Lam = np.zeros(server_num)
        for i in range(server_num):
            for j in range(server_num):
                Lam[j] = Lam[j] + x[i*server_num + j]
        f = 0
        for i in range(server_num):
            for j in range(server_num):
                f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]) + beta_d/x[server_num**2+i*server_num+j])
        return f
    return func

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
    return (server_num, sigma_d, beta_d, sigma_h, beta_h, c)

def check_conditions(conds, x):
    for i in range(len(conds)):
        print("Condition %d, type %s, stage %s, value is: %f" % (i, conds[i]['type'], conds[i]['stage'], conds[i]['fun'](x)))

def gen_conditions(ng):
    
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
    # print("condition matrix is:")
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
    conditions = []
    # conditions for bandwidth of links
    for k in range(l_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[s_len**2:2*s_len**2], C[k,:])
        tmp['stage'] = '1'
        conditions.append(tmp)
        # print("tmp fun:", tmp['fun'](x))
    #conditions for load of offloading
    for i in range(s_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i]
        tmp['stage'] = '2'
        conditions.append(tmp)
        # print("tmp fun:", tmp['fun'](x))

    # # Conditions for load of receiving
    # for j in range(s_len):
        
    #     def gen(j=j):
    #         def load(x):
    #             tmpL = 0
    #             for i in range(s_len):
    #                 tmpL = tmpL + x[i*s_len+j]
    #             return 1/mu[j] - tmpL
    #         return load
    #     tmp = {}
    #     tmp['type'] = 'ineq'
    #     tmp['fun'] = gen(j)
    #     tmp['stage'] = '3'
    #     conditions.append(tmp)
        # print("tmp fun:", tmp['fun'](x))
    
    H = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            H[j, i*s_len + j] = 1
    for j in range(s_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,:])
        tmp['stage'] = '3'
        conditions.append(tmp)

    # Condition for bandwidth of path
    for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[s_len**2+i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN
            tmp['stage'] = '4'
            conditions.append(tmp)
            # print("tmp fun:", tmp['fun'](x))

    # Condition for offloading decision
    for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[i*s_len+j]
            tmp['stage'] = '5'
            conditions.append(tmp)
    
    # print("Conditions:")
    # for c in conditions:
    #     print(c)

    return conditions
    



    # return conditions
    # print(conditions)

def gen_feasible_solution(cons, args):
    print("generating feasible solution:")
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
    server_num, sigma_d, beta_d, sigma_h, beta_h, c = args
    # flag = True
    # num = 1
    # while flag:
    #     print("--------Iteration %d" % (num))

    x0 = np.random.rand(2*server_num**2)

    res = minimize(penalty_func, x0, method='SLSQP',constraints=cons, tol=0.0001)
    print(res)
    # print(res.fun)
    # print(res.x)
    # print(res.message)
    check_conditions(cons, res.x)

    return res.x
    # for i in range(100):
    #     x0 = np.random.rand(2*server_num**2)

    #     res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
        
    #     print(res.success)
    #     print(res.fun)
    #     if res.success:
    #         print(res.x)
    #         check_conditions(cons, res.x)


def reshape_x(x, server_num):
    A = np.zeros((server_num,server_num))
    B = np.zeros((server_num,server_num))

    for i in range(server_num):
        for j in range(server_num):
            A[i,j] = x[i*server_num + j]
            B[i,j] = x[server_num**2 + i*server_num + j]
    return A, B



if __name__ == "__main__":
    # np.random.seed(1)


    tng = createATreeGraph()

    s_list = tng.getServerList() 
    l_list = tng.getLinkList()

    tmp = []
    print("Rsc list:")
    for s in s_list:
        tmp.append(s.getRscAmount())
    print(tmp)
    tmp = []
    print("Ban list:")
    for l in l_list:
        tmp.append(l.getBandwidth())
    print(tmp)

    conditions = gen_conditions(tng)

    x = np.ones(2*len(s_list)**2, float)
    # for i in range(len(conditions)):
    #     print("condition is:", conditions[i])
    #     print("condition %d's value is: %f" % (i,conditions[i]['fun'](x)))
    
    # f = gen_objective(gen_args(tng))
    # print("Objective function:", f(x))
    gen_feasible_solution(conditions, gen_args(tng))
    # y = np.random.rand(2*len(s_list)**2)
    # y[0:len(s_list)**2] = y[0:len(s_list)**2]/1000
    # print("Testing check_conditions:")
    # print("Input is:\n", y)
   
    # check_conditions(conditions, y)
    # # check_conditions(conditions, y)