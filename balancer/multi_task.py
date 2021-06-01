from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from .. import parameters
import numpy as np
from scipy.optimize import minimize
import sys
from matplotlib import pyplot as plt
import json
# from brokenaxes import brokenaxes

PENALTY = 1000000
BETA_D = [0.3,0.2,0.1]
BETA_H = [0.1,0.2,0.3]
TASK_TYPE_LIST = [parameters.CODE_TASK_TYPE_VR, parameters.CODE_TASK_TYPE_VA, parameters.CODE_TASK_TYPE_IoT]
# m is num of servers
# z is num of task types

def gen_m_objective(tng, beta_d, beta_h, task_type_list):
    m = len(tng.getServerList())
    z = len(task_type_list)
    # print("server_num is:", m)
    c = get_rsc_list(tng)
    
    def obj(x):
        lam = gen_lambda(x, m, z)
        prob = gen_prob(x, m, z)
        es = gen_es(prob, beta_h, m)
        ess = gen_ess(prob, beta_h, m)
        f = 0
        for i in range(m):
            for j in range(m):
                for r in range(z):
                    f = f + x[r*m**2 + i*m + j]*(es[j]/c[j] + lam[j]*ess[j]/(2*(c[j]-lam[j]*es[j])))
                    if i != j:
                        f = f + x[r*m**2 + i*m + j]*beta_d[r]/x[z*m**2 + i*m + j]
        return f

    return obj

def gen_es(prob, beta_h, m):
    es_j = np.zeros(m)
    for j in range(m):
        es_j[j] = np.dot(prob[j], beta_h)
    # print("es_j:\n",es_j)
    return es_j

def gen_ess(prob, beta_h, m):
    ess_j = np.zeros(m)
    for j in range(m):
        ess_j[j] = 2*np.dot(prob[j], np.array(beta_h)**2)
    # print("ess_j:\n",ess_j)
    return ess_j

def gen_prob(x, m, z):
    lambda_jr = np.zeros((m,z))
    for j in range(m):
        for r in range(z):
            for i in range(m):
                lambda_jr[j,r] = lambda_jr[j,r] + x[r*m**2+i*m+j]
    lambda_j = gen_lambda(x, m, z)
    prob = np.zeros((m,z))
    for j in range(m):
        prob[j,:] = lambda_jr[j,:]/lambda_j[j]
    # print("prob:\n",prob)
    return prob

def gen_lambda(x, m, z):
    lambda_j = np.zeros(m)
    for i in range(m):
        for j in range(m):
            for r in range(z):
                lambda_j[j] = lambda_j[j] + x[r*m**2+i*m+j]
    # print("lambda:\n", lambda_j)
    return lambda_j

def gen_penalized_obj(obj_func, cons, mode=0):
    penalty_func = gen_penalty_func(cons, mode=mode)
    def penalized_obj(x):
        return penalty_func(x) + obj_func(x)
    return penalized_obj

def gen_penalty_func(cons, mode=0):
    if mode == 0:
        def penalty_func(x):
            f = 0
            for con in cons:
                if con['type'] == 'eq':
                    # pass
                    f = con['fun'](x)**2*PENALTY + f
                elif con['type'] == 'ineq':
                    f = min(0, con['fun'](x))**2*PENALTY + f
            return f
    elif mode == 1:
        def penalty_func(x):
            f = 0
            for con in cons:
                if con['type'] == 'eq':
                    # pass
                    # f = cons['fun'](x)**2*PENALTY + f
                    res = con['func'](x)
                    print("Result of condithon:")
                    for r in res:
                        f = f + r**2*PENALTY
                elif cons['type'] == 'ineq':
                    res = con['fun'](x)
                    for r in res:
                        f = min(0, r)**2*PENALTY + f
            return f
    return penalty_func

# Generating constraints.
def gen_m_constraints(tng, beta_d, beta_h, task_type_list, mode=0):
    alpha = get_alpha(tng, task_type_list=task_type_list)
    c = get_rsc_list(tng)
    # print("beta_d:", beta_d)
    # print("beta_h:", beta_h)
    constraints = []
    m = len(tng.getServerList())
    z = len(task_type_list)
    s_list = tng.getServerList()
    l_list = tng.getLinkList()

    # Constraint for sender's load
    C1 = np.zeros((m*z,z*m**2 + m**2))
    b1 = np.zeros(m*z)
    info1 = []
    type1 = []
    for i in range(m):
        for r in range(z):
            b1[i*z + r] = alpha[i,r]
            info1.append("remaining load of task %s in %s" % (task_type_list[r], s_list[i].getKey()))
            type1.append('eq')
            for j in range(m):
                C1[i*z + r, r*m**2 + i*m + j] = -1

    # Constraint for receiver's load
    C2 = np.zeros((m, z*m**2 + m**2))
    b2 = np.zeros(m)
    info2 = []
    type2 = []
    for j in range(m):
        b2[j] = c[j]
        info2.append('remaining rsc of %s' % (s_list[j].getKey()))
        type2.append('ineq')
        for i in range(m):
            for r in range(z):
                C2[j,r*m**2 + i*m + j] = - beta_h[r]
    
    # Constraint for link's load
    C3 = np.zeros((len(l_list), z*m**2 + m**2))
    b3 = np.zeros(len(l_list))
    info3 = []
    type3 = []
    for k in range(len(l_list)):
        b3[k] = l_list[k].getBandwidth()
        info3.append('remaining bandwith of %s ' % (l_list[k].getKey()))
        type3.append('ineq')
        for i in range(m):
            for j in range(m):
                if l_list[k] in tng.getShortestPath(s_list[i], s_list[j]).getLinkList():
                    C3[k,z*m**2 + i*m + j] = -1
    
    # Constraint for bandwidth of path:
    C4 = np.zeros((m**2, z*m**2 + m**2))
    b4 = np.zeros(m**2)
    info4 = []
    type4 = []
    for i in range(m):
        for j in range(m):
            C4[i*m +j, z*m**2 + i*m +j] = 1
            info4.append('surplus bandwith of path from %s to %s' %(s_list[i].getKey(), s_list[j].getKey()) )
            type4.append('ineq')
            for r in range(z):
                C4[i*m + j, r*m**2 + i*m + j] = - beta_d[r]
    
    # Constraint for offloading decison
    C5 = np.zeros((z*m**2, z*m**2 + m**2))
    b5 = np.zeros(z*m**2)
    info5 = []
    type5 = []
    for i in range(m):
        for j in range(m):
            for r in range(z):
                info5.append('offloading of task %s from %s to %s' %(task_type_list[r],s_list[i].getKey(), s_list[j].getKey()))
                type5.append('ineq')
                C5[r*m**2 + i*m + j, r*m**2 + i*m + j] = 1

    if mode == 0:
        cons = []
        C = np.concatenate((C1,C2,C3,C4,C5), axis=0)
        b = np.concatenate((b1,b2,b3,b4,b5))
        info = info1 + info2 + info3 + info4 + info5
        _type = type1 + type2 + type3 + type4 + type5
        length = len(b)
        for i in range(length):
            tmp = {}
            tmp['type'] = _type[i]
            tmp['fun'] = lambda x, i=i: np.dot(x, C[i]) + b[i]
            tmp['info'] = info[i]
            cons.append(tmp)
    elif mode == 1:
        cons = []
        Cs = [C1, C2, C3, C4, C5]
        bs = [b1, b2, b3, b4, b5]
        types = [type1, type2, type3, type4, type5]
        info = [info1, info2, info3, info4, info5]
        for i in range(len(bs)):
            tmp = {}
            tmp['type'] = types[i][0]
            tmp['fun'] = lambda x, c=Cs[i], b=bs[i]: np.dot(x, c) + b
            tmp['info'] = info[i]
            cons.append(tmp)
    
    return cons            

def check_m_constraints(cons, x, mode=0):
    if mode == 0:
        for i in range(len(cons)):
            print("Constraint %d, type %s, %s is: %f" % (i, cons[i]['type'], cons[i]['info'], cons[i]['fun'](x)))
    elif mode == 1:
        num = 0
        for e in cons:
            info = e['info']
            _type = e['type']
            res = e['fun'](x)
            for i in range(len(res)):
                print("Constraint %d, type %s, %s is: %f" % (num, _type, info[i], res[i]))
                num = num + 1

# Solvers
def solve_multi(tng, beta_d, beta_h, task_type_list):
    obj = gen_m_objective(tng, beta_d=beta_d, beta_h=beta_h, task_type_list=task_type_list)
    cons = gen_m_constraints(tng, beta_d=beta_d, beta_h=beta_h, task_type_list=task_type_list)
    m = len(tng.getServerList())
    z = len(task_type_list)

    iteration=20
    success = 0
    for i in range(iteration):
        x0 = gen_m_feasible_solution(cons, tng,task_type_list=task_type_list, true_initial=True).x
        res = minimize(obj, x0, method='SLSQP',constraints=cons)
        if res.success == True:
            success = success + 1
            print(res)
            check_m_constraints(cons, res.x)
            print_result(res.x, m, z, task_type_list=task_type_list)
    print("Iterate %d times, succeed %d times, ratio %f" % (iteration, success, success/iteration))

def test_multi(tng, beta_d, beta_h, task_type_list, repeat=5):
    obj = gen_m_objective(tng, beta_d=beta_d, beta_h=beta_h, task_type_list=task_type_list)
    cons = gen_m_constraints(tng, beta_d=beta_d, beta_h=beta_h, task_type_list=task_type_list)
    m = len(tng.getServerList())
    z = len(task_type_list)

    iteration=20
    success = 0
    min_fun =1000000000
    min_res = None
    while success < repeat:
        x0 = gen_m_feasible_solution(cons, tng,task_type_list=task_type_list, true_initial=True).x
        res = minimize(obj, x0, method='SLSQP',constraints=cons)
        if res.success == True and res.fun < 1000:
            success = success + 1
            print("---------success count %d---------" % success)
            print(res.fun)
            # check_m_constraints(cons, res.x)
            # print_result(res.x, m, z, task_type_list=task_type_list)
            # parse_result(tng=tng, x = res.x, z = z)
            if res.fun < min_fun:
                min_fun = res.fun
                min_res = res
    # return min_res.x
    # tng.print_info()
    # print(min_res)
    check_m_constraints(cons, min_res.x)
    # print_result(min_res.x, m, z, task_type_list=task_type_list)
    return parse_result(tng=tng, x=min_res.x, z=z)
    
def parse_result(tng, x, z):
    """ 进行一些统计，将问题的解，解析为我们需要的形式 """
    m = len(tng.getServerList())
    x = np.reshape(x, ((z+1)*m, m))
    groups = tng.getGroupList()
    poops = np.zeros((len(groups), z))
    s_list = tng.getServerList()
    for v in range(len(groups)):
        indicators = np.zeros(m)
        group = groups[v]
        servers_in_group = tng.getServersInGroup(group)
        # 解析出域内边缘服务器对应到解中的行号
        
        for k in range(len(servers_in_group)):
            # print(servers_in_group[k].getKey())
            for i in range(len(s_list)):
                if s_list[i].getKey() == servers_in_group[k].getKey():
                    indicators[i] = 1
                    break
        # print("indicators for %s" % group.getKey())
        # print(indicators)
        for r in range(z):
            # result_for_type = x[r*m:(r+1)*m,:]
            receiver_loads = np.sum(x[r*m:(r+1)*m,:], axis=0)
            # print(receiver_loads)
            poops[v, r] = np.dot(receiver_loads, indicators)
            # print("Offloading decisions for task %s:" % task_type_list[i])
            # print(x[i*m:(i+1)*m,:])
            # print("Senders offloaded:\n", np.sum(x[i*m:(i+1)*m,:], axis=1))
            # print("Receivers received:\n", np.sum(x[i*m:(i+1)*m,:], axis=0))
        # print("Bandwidth for pathes:\n", x[(z-1)*m:z*m, :])
    print("parsed result:\n", poops)
    return poops

# Generating feasible solution
def gen_m_feasible_solution(cons, tng, task_type_list, true_initial=False,mode=0):
    penalty_func = gen_penalty_func(cons, mode=mode)
    m = len(tng.getServerList())
    z = len(task_type_list)
    if true_initial == True:
        for i  in range(100):
            x0 = np.random.rand((z+1)*m**2)
            res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
            if res.success ==True:
                break
        if i==100:
            print("Iteration for finding initial value exceeds 100 times.")
            sys.exit(10)
    else:
        x0 = np.random.rand((z+1)*m**2)
        res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
    # print(res)
    # check_m_constraints(cons, res.x)
    print(res.success)

    return res
def linear_obj(x):
    return sum(x)

# Some helpful tools
def get_rsc_list(tng):
    c = []
    s_list = tng.getServerList()
    for s in s_list:
        c.append(s.getRscAmount())
    # print("beta_d:\n",_beta_d)
    # print("beta_h",_beta_h)
    print("Servers' rsc:\n", c)
    # print("np.dot(beta_d, beta_h)",np.dot(_beta_d, _beta_h))
    return c

def get_alpha(tng, task_type_list):
    s_list = tng.getServerList()
    m = len(s_list)
    z = len(task_type_list)
    alpha = np.zeros((m, z))
    for i in range(m):
        for r in range(z):
            alpha[i,r] = s_list[i].getGroup().getTaskGenInfo(task_type_list[r])[0]
    # print("alpha:\n",alpha)
    return alpha

def print_result(x, m, z, task_type_list):
    # print(x)
    x = np.reshape(x, ((z+1)*m, m))
    for i in range(z):
        print("Offloading decisions for task %s:" % task_type_list[i])
        print(x[i*m:(i+1)*m,:])
        print("Senders offloaded:\n", np.sum(x[i*m:(i+1)*m,:], axis=1))
        print("Receivers received:\n", np.sum(x[i*m:(i+1)*m,:], axis=0))
    print("Bandwidth for pathes:\n", x[(z-1)*m:z*m, :])


def test_tools(tng, beta_d, beta_h, task_type_list):
    get_alpha(tng)
    get_rsc_list(tng)
    m = len(tng.getServerList())
    z = len(task_type_list)

    cons = gen_m_constraints(tng)
    # x = np.ones(((z+1)*m**2))
    # check_m_constraints(cons, x)
    # x= np.random.rand(((z+1)*m**2))
    # print_result(x, m, z)
    res = gen_m_feasible_solution(cons, tng, true_initial=True)
    print_result(res.x, m, z)
    lam = gen_lambda(res.x, m, z)
    prob = gen_prob(res.x, m, z)
    es = gen_es(prob, beta_h=beta_h, m=m)
    ess = gen_ess(prob, beta_h=beta_h, m=m)
    print("load:\n", lam*es)


if __name__=='__main__':
    np.set_printoptions(formatter={'float':'{:.3f}'.format})
    tng = createATreeGraph()
    tng.print_info()

    # test_multi_types(tng, [1,2,3], [4,5,6])
    solve_multi(tng, beta_d=BETA_D, beta_h=BETA_H, task_type_list=TASK_TYPE_LIST)
    # test_tools(tng)
