from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from ..parameters import *
import numpy as np
from scipy.optimize import minimize
import sys
from matplotlib import pyplot as plt

# Control task model here
VA_COM_MEAN = 3
VA_COM_VAR = 1
VA_DATA_MEAN = 1
VA_DATA_VAR = 1
ADJUST = 0.0001

# Control algorithm parameters here
PENALTY = 1000000


def gen_objective(args, mode=0, argA=None, argB=None):
    server_num, sigma_d, beta_d, sigma_h, beta_h, c = args
    mu = np.zeros(server_num)
    omega = np.zeros(server_num)

    # if mode == 1 and argA == None:
    #     print("If mode is 1, then argA should not be None")
    #     sys.exit(5)
    # elif mode == 2 and argB == None:
    #     print("If modd is 2, then argB should not be None")
    #     sys.exit(5)

    for j in range(server_num):
        mu[j] = c[j]/beta_h
        omega[j] = ((sigma_h/c[j])**2*mu[j]**2+1)/(2*mu[j])
    if mode == 0:
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    if i != j:
                        f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]) + beta_d/x[server_num**2+i*server_num+j])
            return f
    elif mode == 1:
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + argA[i*server_num + j]
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    if i != j:
                        f = f + argA[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]) + beta_d/x[i*server_num+j])
                        # f = f + argA[i*server_num+j]*beta_d/x[i*server_num+j]
            return f
    elif mode == 2:
        def func(x):
            Lam = np.zeros(server_num)
            for i in range(server_num):
                for j in range(server_num):
                    Lam[j] = Lam[j] + x[i*server_num + j]
            f = 0
            for i in range(server_num):
                for j in range(server_num):
                    if i != j:
                        f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]) + beta_d/argB[i*server_num+j])
                        # f = f + x[i*server_num+j]*(1/mu[j] + omega[j]*Lam[j]/(mu[j]-Lam[j]))
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
    if mode == 0 or mode == 2:
        for i in range(s_len):
            tmp = {}
            tmp['type'] = 'eq'
            tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i] - ADJUST
            tmp['info'] = 'sended load of %s' % (s_list[i].getKey())
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
    if mode == 0 or mode==2:
        for i in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i] - ADJUST
            tmp['info'] = 'sended load of %s' % (s_list[i].getKey())
            constraints.append(tmp)
    elif mode == 1:
        constraints = []
    return constraints

def constraints_for_receiver(ng, mode=0, argA=None, argB=None):
    s_list, l_list, s_len, l_len, alpha, mu = get_ng_info_for_constraints(ng)
    # print('mu:')
    print(mu)
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
    elif mode == 2:
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,0:s_len**2])
            tmp['info'] = 'received load of %s' % (s_list[j].getKey())
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
                tmp['fun'] = lambda x, i=i, j=j, argB=argB: argB[i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN - ADJUST
                tmp['info'] = 'surplus bandwith of path from %s to %s' %(s_list[i].getKey(), s_list[j].getKey()) 
                constraints.append(tmp)
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
    if mode == 0 or mode ==2:
        for i in range(s_len):
            for j in range(s_len):
                tmp = {}
                tmp['type'] = 'ineq'
                tmp['fun'] = lambda x, i=i, j=j: x[i*s_len+j] # - ADJUST
                tmp['info'] = 'offloading from %s to %s' %(s_list[i].getKey(), s_list[j].getKey())
                constraints.append(tmp)
    
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

def gen_constraints(ng):
    
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
    for k in range(l_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, k=k: bandwidth[k] - np.dot(x[s_len**2:2*s_len**2], C[k,:])
        tmp['info'] = 'link'
        constraints.append(tmp)
        # print("tmp fun:", tmp['fun'](x))
    #constraints for load of offloading
    for i in range(s_len):
        tmp = {}
        tmp['type'] = 'eq'
        tmp['fun'] = lambda x, i=i: np.sum(x[i*s_len:(i+1)*s_len]) - alpha[i]
        tmp['info'] = 'sender'
        constraints.append(tmp)
        # print("tmp fun:", tmp['fun'](x))

    # # constraints for load of receiving
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
    #     tmp['info'] = 'receiver'
    #     constraints.append(tmp)
        # print("tmp fun:", tmp['fun'](x))
    
    H = np.zeros((s_len, 2*s_len**2))
    for i in range(s_len):
        for j in range(s_len):
            H[j, i*s_len + j] = 1
    for j in range(s_len):
        tmp = {}
        tmp['type'] = 'ineq'
        tmp['fun'] = lambda x, j=j: mu[j] - np.dot(x, H[j,:])
        tmp['info'] = 'receiver'
        constraints.append(tmp)

    # Constraint for bandwidth of path
    for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[s_len**2+i*s_len+j] - x[i*s_len+j]*VA_DATA_MEAN
            tmp['info'] = 'path'
            constraints.append(tmp)
            # print("tmp fun:", tmp['fun'](x))

    # Constraint for offloading decision
    for i in range(s_len):
        for j in range(s_len):
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = lambda x, i=i, j=j: x[i*s_len+j]
            tmp['info'] = 'offloading'
            constraints.append(tmp)
    
    # print("constraints:")
    # for c in constraints:
    #     print(c)

    return constraints
    



    # return constraints
    # print(constraints)

def gen_feasible_solution_exterior(cons, args):
    print("generating feasible solution by exterior_penalty:")
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
    # print(res)
    # print(res.fun)
    # print(res.x)
    # print(res.message)
    # check_constraints(cons, res.x)
    print(res.success)

    return res
  


def reshape_x(x, server_num):
    A = np.zeros((server_num,server_num))
    B = np.zeros((server_num,server_num))

    for i in range(server_num):
        for j in range(server_num):
            A[i,j] = x[i*server_num + j]
            B[i,j] = x[server_num**2 + i*server_num + j]
    return A, B

def interior_penalty(cons):
    def func(x):
        f = 0
        for e in cons:
            if e['type'] == 'ineq':
                f = f + e['fun'](x)
            else:
                print("interior can't handle equation constraint.")
                sys.exit(1)
        return f
    return func

def exterior_penalty(cons):
    def func(x):
        f = 0
        for e in cons:
            if e['type'] == 'eq':
                f = f + e['fun']**2*PENALTY
            elif e['type'] == 'ineq':
                f = f + max(0, -e['fun'])**2*PENALTY
            else:
                print("Unknow type of constraint, type name is %s" % (e['type']))
                sys.exit(2)
        return f
    return func


def gen_feasible_solution_interior(cons, args):
    print("Generating feasible solution by interior_penalty:")
    in_func = interior_penalty(cons)
    server_num, sigma_d, beta_d, sigma_h, beta_h, c = args
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
    np.random.seed(5)
    for i in range(iteration):
        res = gen_feasible_solution_exterior(constraints2, gen_args(ng))
        if res.success:
            success_ex = success_ex + 1
    print("Exterior penalty, iteration %d, success %d, ratio %f" % (iteration, success_ex, success_ex/iteration))

    print("Test interior:")
    np.random.seed(5)
    for i in range(iteration):
        res = gen_feasible_solution_interior(constraints2, gen_args(ng))
        if res.success:
            success_in = success_in + 1
    print("Interior penalty, iteration %d, success %d, ratio %f" % (iteration, success_in, success_in/iteration))

def test_constraints_division_whether_equal(tng):
    """ If we divide constraints into five, are they equal in function? """
    constraints = gen_constraints(tng)

    constraints2 = constraints_for_links(tng)
    constraints2 = constraints_for_sender_var(tng) + constraints2
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
    res = gen_feasible_solution_interior(constraints, gen_args(tng))
    print(res)
    print("for constraints two:")
    res = gen_feasible_solution_interior(constraints2, gen_args(tng))
    print(res)

def test_partial_iteration(tng):
    # Doesn't work well, or  pretty bad.
    np.random.seed()
    constraints1 = constraints_for_links(tng)
    constraints1 = constraints_for_sender_var(tng) + constraints1
    constraints1 = constraints_for_receiver(tng) + constraints1
    constraints1 = constraints_for_path_ban(tng) + constraints1
    constraints1 = constraints_for_offloading(tng) + constraints1

    args = gen_args(tng)
    x0 = gen_feasible_solution_exterior(constraints1, args).x
    # x0 = np.random.rand(2*len(s_list)**2)
    argA = x0[0:len(s_list)**2]
    argB = x0[len(s_list)**2: 2*len(s_list)**2]

    record_fun = []
    record_f = []
    iteration = 200

    # cons = []
    # cons = constraints_for_links(tng, mode=1, argA=argA)
    # cons = constraints_for_offloading(tng, mode=1, argA=argA) + cons
    # cons = constraints_for_path_ban(tng, mode=1, argA=argA) + cons
    # cons = constraints_for_receiver(tng, mode=1, argA=argA) + cons
    # cons = constraints_for_sender(tng, mode=1, argA=argA) + cons
    f = gen_objective(args)
    # argA = np.diag([1,1,1,1])
    # argA = np.reshape(argA,(16))
    # print("argA:\n",argA)
    # eps = 0.001


    for i in range(iteration):
    
        cons = []
        cons = constraints_for_links(tng, mode=2, argB=argB)
        cons = constraints_for_offloading(tng, mode=2, argB=argB) + cons
        cons = constraints_for_path_ban(tng, mode=2, argB=argB) + cons
        cons = constraints_for_receiver(tng, mode=2, argB=argB) + cons
        cons = constraints_for_sender(tng, mode=2, argB=argB) + cons
        resA = minimize(gen_objective(args, mode=2, argB=argB), argA, method='SLSQP',constraints=cons, options={"maxiter":1})# , options={"maxiter":1})
        argA = resA.x
        record_fun.append(resA.fun)
        record_f.append(f(np.concatenate((np.array(argA),np.array(argB)))))

        # check_constraints(cons, resA.x)
        # print(resA)
        

        
        
        cons = []
        cons = constraints_for_links(tng, mode=1, argA=argA)
        cons = constraints_for_offloading(tng, mode=1, argA=argA) + cons
        cons = constraints_for_path_ban(tng, mode=1, argA=argA) + cons
        cons = constraints_for_receiver(tng, mode=1, argA=argA) + cons
        cons = constraints_for_sender(tng, mode=1, argA=argA) + cons
        resB = minimize(gen_objective(args, mode=1, argA=argA), argB, method='SLSQP',constraints=cons, options={"maxiter":1})# options={"maxiter":1})
        argB = resB.x
        record_fun.append(resB.fun)
        record_f.append(f(np.concatenate((np.array(argA),np.array(argB)))))
        # check_constraints(cons, resB.x)
        print("Offloading:")
        print(np.reshape(argA,(len(s_list), len(s_list))))
        print("Bandwidth:")
        print(np.reshape(argB,(len(s_list), len(s_list))))
        # print(resB)
    # print("Iteration history is:")
    # print(record_fun)
    # print(range(60))
    x = np.arange(1,iteration*2+1)
    plt.title("Matplotlib demo") 
    plt.xlabel("x axis caption") 
    plt.ylabel("y axis caption") 
    plt.plot(x, record_fun)
    plt.plot(x, record_f)
    plt.show()

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
    

if __name__ == "__main__":
    # np.random.seed(1)
    np.set_printoptions(formatter={'float':'{:.3f}'.format})

    tng = createATreeGraph()
    tng.print_info()

    s_list = tng.getServerList() 
    l_list = tng.getLinkList()


    # test_objective_function(tng)
    test_partial_iteration(tng)
    # test_constraints_division_whether_equal(tng)
    # test_efficiency_of_penalty_funcions(tng)