from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from .. import parameters
import numpy as np
from scipy.optimize import minimize
import sys
from matplotlib import pyplot as plt
import json
from .gen_problem import print_result, check_constraints
# from brokenaxes import brokenaxes

PENALTY = 1000000
BETA_D = [2,0.2,0.1]
SIGMA_D = [1,1,1]
BETA_H = [0.5,0.2,0.3]
SIGMA_H = [1,1,1]
TASK_TYPE_LIST = [parameters.CODE_TASK_TYPE_VR, parameters.CODE_TASK_TYPE_VA, parameters.CODE_TASK_TYPE_IoT]


def gen_obj(tng, max_iteration=100, mode=0):
    # First, extract information
    # Backhaul link
    # Second, construct obj func
    def obje(x):
        g_list = tng.getGroupList()
        group_objs = []
        group_cons = []
        r = len(g_list)
        print("x is:\n", x)
        for g in g_list:
            # print("generating group opt problem, %s" % g.getKey())
            group_objs.append(gen_group_obj(tng, g))
            group_cons.append(gen_group_constraints(tng, g, x))
        f_g_v = []
        for i in range(len(g_list)):
            success = False
            count = 0
            # while success == False:
            g = g_list[i]
            cons = gen_group_constraints(tng, g, x)
            while success == False and count < max_iteration:
                
                
                try:
                    x0 =  gen_group_feasible_solution(cons, tng, g, x, True).x
                
                    res = minimize(group_objs[i], x0, method='SLSQP',constraints=group_cons[i])
                    fun = res.fun
                    count = count + 1
                    success = res.success
                except ValueError:
                    return 100

            f_g_v.append(fun)

        f = 0
        for u in range(r): 
            for v in range(r):
                # print("f_g_v[v]",f_g_v[v])
                # f = f + x[u*r + v]*f_g_v[v]
                f = f + f_g_v[v]
                if u != v and x[r**2 + u*r + v] > 0:
                    f = f + x[u*r + v]*BETA_D[0]/x[r**2 + u*r + v]
                    # print("x[u*r + v]*BETA_D[0]/x[r**2 + u*r + v", x[u*r + v]*BETA_D[0]/x[r**2 + u*r + v])
        # print("f is: ", f)
        return f

    def obj2(x):
        return np.random.rand()*10
    if mode == 0:
        return obje
    elif mode == 1:
        return obj2

    # return obje

def gen_group_obj(tng, group, mode='full'):
    # print("In gen_group_obj, %s" % group.getKey())
    s_list = tng.getServersInGroup(group)
    m = len(s_list)
    l_list = tng.getLinksInGroup(group)
    l_len = len(l_list)
    beta_d, beta_h, c = gen_group_args(tng, group)
    mu = gen_mu(tng, group)
    sigma = gen_sigma(tng, group)
    if mode=='full':
        def obj(x):
            # print("what happend?")
            lam = gen_group_lambda(x, m)
            f = 0
            for j in range(m):
                com_time = 1/mu[j] + (sigma[j]**2*mu[j]**2 + 1)/(2*mu[j])*lam[j]/max(0.0001,(mu[j] - lam[j]))
                # print("com_time:",com_time) 
                f = f + x[j]*com_time
                for i in range(m):
                    f = f + x[2*m + i*m + j]*com_time
                    if i != j and x[2*m + m**2 + i*m + j] > 0:
                        f = f + x[2*m + i*m + j]*beta_d[0]/x[2*m + m**2 + i*m + j]
            return f
    
    elif mode=='com':
        def obj(x):
            lam = gen_group_lambda(x, m)
            f = 0
            for j in range(m):
                com_time = 1/mu[j] + (sigma[j]**2*mu[j]**2 + 1)/(2*mu[j])*lam[j]/max(0.0001,(mu[j] - lam[j]))
                f = f + x[j]*com_time
                for i in range(m):
                    f = f + x[2*m + i*m + j]*com_time
            return f
    elif mode=='trans':
        def obj(x):
            lam = gen_group_lambda(x, m)
            f = 0
            for j in range(m):
                for i in range(m):
                    if i != j and x[2*m + m**2 + i*m + j]>0:
                        # print("x[2*m + i*m + j],beta_d[0],x[2*m + m**2 + i*m + j]: ", x[2*m + i*m + j],beta_d[0],x[2*m + m**2 + i*m + j])
                        f = f + x[2*m + i*m + j]*beta_d[0]/x[2*m + m**2 + i*m + j]
            return f
    else:
        print("No mode named %s" % str(mode))
    return obj

def gen_mu(tng, group):
    s_list = tng.getServersInGroup(group)
    mu = np.zeros(len(s_list))
    for i in range(len(s_list)):
        mu[i] = s_list[i].getRscAmount()/BETA_H[0]
    # print("mu for servers in %s" % group.getKey())
    # print(mu)
    return mu

def gen_sigma(tng, group):
    s_list = tng.getServersInGroup(group)
    sigma = np.zeros(len(s_list))
    for i in range(len(s_list)):
        sigma[i] = SIGMA_H[0]/s_list[i].getRscAmount()
    # print('sigma:\n',sigma)
    return sigma

def gen_group_lambda(x, m):
    lam = np.zeros(m)
    for j in range(m):
        lam[j] = lam[j] + x[j]
        for i in range(m):
            lam[j] = lam[j] + x[2*m+i*m+j]
    # print("lambda:\n", lam)
    return lam

def gen_obj_v2(tng):
    pass
    

# Constraints for group offloading
def gen_group_constraints(tng, group, x_g):
    # print("In gen_group_constraints")
    g_list = tng.getGroupList()
    g_index = -1
    for i in range(len(g_list)):
        if g_list[i].getKey() == group.getKey():
            g_index = i
            break
    if g_index == -1:
        print("Something is wrong with your group list")
        sys.exit()

    s_list = tng.getServersInGroup(group)
    l_list = tng.getLinksInGroup(group)
    m = len(s_list)
    # print("In %s" % group.getKey())
    gamma_v, theta_u, phi_v_in, phi_v_out = gen_inter_data(x_g, g_index, len(g_list))
    mu = np.zeros(m)
    alpha = np.zeros(m)
    for i in range(m):
        mu[i] = s_list[i].getRscAmount()/BETA_H[0]
        alpha[i] = s_list[i].getGroup().getTaskGenInfo(TASK_TYPE_LIST[0])[0]
    # print("mu:\n",mu)
    # print("alpha:\n",alpha)
    # Constraint for group input, output
    length = 2*m + 2*m**2
    C1 = np.zeros((2,length))
    b1 = np.zeros(2)
    type1 = ['eq', 'eq']
    info1 = ['remaining input of %s' % (group.getKey()), 'remaining output of %s' % (group.getKey())]
    for j in range(m):
        C1[0, j] = -1
        C1[1, m + j] = -1
    b1[0] = gamma_v
    b1[1] = theta_u
    
    # Constraint for sender's load
    C2 = np.zeros((m, length))
    b2 = np.zeros(m)
    type2 = []
    info2 = []
    for i in range(m):
        C2[i, m + i] = -1
        b2[i] = alpha[i]
        type2.append('eq')
        info2.append("remaning load of %s" % s_list[i].getKey())
        for j in range(m):
            C2[i, 2*m + i*m + j] = -1
    
    # Constraint for receiver's load
    C3 = np.zeros((m, length))
    b3 = np.zeros(m)
    type3 = []
    info3 = []
    for j in range(m):
        C3[j, j] = -1
        b3[j] = mu[j]
        type3.append('ineq')
        info3.append("remaining rsc of %s" % s_list[j].getKey())
        for i in range(m):
            C3[j, 2*m + i*m + j] = -1

    # Constraint for path ban
    C4 = np.zeros((m**2, length))
    b4 = np.zeros(m**2)
    type4 = []
    info4 = []
    for i in range(m):
        for j in range(m):
            type4.append('ineq')
            info4.append("surplus bandwidth of path from %s to %s" % (s_list[i].getKey(), s_list[j].getKey()))
            C4[i*m + j, 2*m + m**2 + i*m + j] = 1
            C4[i*m + j, 2*m + i*m + j] = -BETA_D[0]
    
    # Constraint for link ban
    C5 = np.zeros((len(l_list), length))
    b5 = np.zeros(len(l_list))
    type5 = []
    info5 = []
    for k in range(len(l_list)):
        b5[k] = l_list[k].getBandwidth()
        type5.append('ineq')
        info5.append("remaining bandwidth of %s" % l_list[k].getKey())
        for i in range(m):
            if l_list[k].getHeadAP().getKey() == s_list[i].getKey():
                if theta_u > 0.001:
                    C5[k, m + i] =  -phi_v_out/theta_u
                else:
                    C5[k, m + i] = 0
                # print("-phi_v_out/theta_u",-phi_v_out/theta_u)
            elif l_list[k].getTailAP().getKey() == s_list[i].getKey():
                if gamma_v > 0.001:
                    C5[k, i] = -phi_v_in/gamma_v
                else:
                    C5[k, i] = 0
                # print("-phi_v_in/gamma_v",-phi_v_in/gamma_v)
            for j in range(m):
                if l_list[k] in tng.getShortestPath(s_list[i], s_list[j]).getLinkList():
                    C5[k, 2*m + m**2 + i*m + j] = -1

    # Constraint for offloading decision
    C6 = np.zeros((m**2, length))
    b6 = np.zeros(m**2)
    type6 = []
    info6 = []
    for i in range(m):
        for j in range(m):
            type6.append('ineq')
            info6.append("offloading from %s to %s" % (s_list[i].getKey(), s_list[j].getKey()))
            C6[i*m + j, 2*m + i*m + j] = 1

    # Constraint for input and output decision
    C7 = np.zeros((2*m, length))
    b7 = np.zeros(2*m)
    type7 = []
    info7 = []
    for i in range(m):
        type7.append('ineq')
        info7.append("input decision of %s" % s_list[i].getKey())
        C7[i, i] = 1
        type7.append('ineq')
        info7.append("output decision of %s" % s_list[i].getKey())
        C7[m + i, m + i] = 1
    
    C = np.concatenate((C1, C2, C3, C4, C5, C6, C7), axis=0)
    b = np.concatenate((b1, b2, b3, b4, b5, b6, b7))
    info = info1 + info2 + info3 + info4 + info5 + info6 + info7
    _type = type1 + type2 + type3 + type4 + type5 + type6 + type7
    cons = []
    for i in range(len(b)):
        tmp = {}
        tmp['type'] = _type[i]
        tmp['info'] = info[i]
        tmp['fun'] = lambda x, i=i: np.dot(x, C[i]) + b[i]
        cons.append(tmp)
    
    return cons

# Constraints for inter-group offloading
def gen_constraints(tng):
    alpha_g = gen_alpha_g(tng)
    mu_g = gen_mu_g(tng)
    g_list = tng.getGroupList()
    r = len(g_list)

    # Constraint for sender's load
    C1 = np.zeros((r, 2*r**2))
    b1 = np.zeros(r)
    type1 = []
    info1 = []
    # print("alpha_g:\n",alpha_g)
    # print("mu_g:\n", mu_g)
    for u in range(r):
        b1[u] = alpha_g[u]
        type1.append('eq')
        info1.append('remaining load of sender %s' % g_list[u].getKey())
        for v in range(r):
            C1[u, u*r + v] = -1
    
    # Constraint for receiver's load
    C2 = np.zeros((r, 2*r**2))
    b2 = np.zeros(r)
    type2 = []
    info2 = []
    for v in range(r):
        b2[v] = mu_g[v]
        type2.append('ineq')
        info2.append('remaining rsc of %s' % g_list[v].getKey())
        for u in range(r):
            C2[v, u*r + v] = -1
    
    # Constraint for path ban
    C3 = np.zeros((r**2, 2*r**2))
    b3 = np.zeros(r**2)
    type3 = []
    info3 = []
    for u in range(r):
        for v in range(r):
            type3.append('ineq')
            info3.append('surplus bandwidth of path from %s to %s' % (g_list[u].getKey(), g_list[v].getKey()))
            C3[u*r + v, r**2 + u*r + v] = 1
            C3[u*r + v, u*r + v] = -BETA_D[0]

    # Constraint for link's load
    l_list = tng.get_backhaul_links()
    C4 = np.zeros((len(l_list), 2*r**2))
    b4 = np.zeros(len(l_list))
    type4 = []
    info4 = []
    for k in range(len(l_list)):
        b4[k] = l_list[k].getBandwidth()
        type4.append('ineq')
        info4.append('remaining bandwidth of %s' % l_list[k].getKey())
        for u in range(r):
            for v in range(r):
                if l_list[k] in tng.getShortestPath(tng.getSwitchInGroup(g_list[u]), tng.getSwitchInGroup(g_list[v])).getLinkList():
                    C4[k, r**2 + u*r + v] = -1
    
    # Constraint for offloading decision
    C5 = np.zeros((r**2, 2*r**2))
    b5 = np.zeros((r**2))
    type5 = []
    info5 = []
    for u in range(r):
        for v in range(r):
            type5.append('ineq')
            info5.append('offloading from %s to %s' % (g_list[u].getKey(), g_list[v].getKey()))
            C5[u*r + v, u*r + v] = 1

    C = np.concatenate((C1, C2, C3, C4, C5), axis=0)
    b = np.concatenate((b1, b2, b3, b4, b5))
    types = type1 + type2 + type3 + type4 + type5
    infos = info1 + info2 + info3 + info4 + info5
    cons = []
    for i in range(len(b)):
        tmp = {}
        tmp['type'] = types[i]
        tmp['info'] = infos[i]
        tmp['fun'] = lambda x, i=i: np.dot(x, C[i]) + b[i]
        cons.append(tmp)
    
    return cons

def gen_alpha_g(tng):
    # print("gen_alpha_g")
    alpha_g = []
    g_list = tng.getGroupList() 
    # for r in range(len(g_list)):
    #     print("group:", g_list[r].getKey())
    for r in range(len(g_list)):
        alpha_g_r = 0
        print("In %s" % g_list[r].getKey())
        for s in tng.getServersInGroup(g_list[r]):
            # print("%s has load %f" % (s.getKey(), s.getGroup().getTaskGenInfo(TASK_TYPE_LIST[0])[0]))
            if s.isServer():
                alpha_g_r = alpha_g_r + s.getGroup().getTaskGenInfo(TASK_TYPE_LIST[0])[0]
            else:
                print("Something is wrong with your tng methods")
                sys.exit()
        alpha_g.append(alpha_g_r)
    alpha_g =  np.array(alpha_g)
    print("alpha_g:\n", alpha_g)
    return alpha_g

def gen_mu_g(tng):
    g_list = tng.getGroupList()
    mu_g = np.zeros(len(g_list))
    for r in range(len(g_list)):
        c_r = 0
        for s in tng.getServersInGroup(g_list[r]):
            c_r = c_r + s.getRscAmount()
        # print("rsc in %s" % g_list[r].getKey())
        # print(c_r)
        mu_g[r] = c_r/BETA_H[0]
    print("mu_g", mu_g)
    return mu_g


# Generating feasible solution
def gen_group_feasible_solution(cons, tng, group,x_g, true_initial=False):
    # print("In gen_group_feasible_solution")
    m = len(tng.getServersInGroup(group))
    penalty_func = gen_penalty_func(cons)
    # x = np.ones(2*m + 2*m**2)
    # check_constraints(cons, x)
    if true_initial == True:
        # print("true_initial:", true_initial)
        for i  in range(200):
            # print("iteration ", i)
            x0 = np.random.rand(2*m + 2*m**2)
            res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
            # print(res)
            # check_constraints(cons, res.x)
            if res.success ==True:
                break
        if i==199:
            print("Iteration for finding initial value in %s exceeds 100 times." % group.getKey())
            print(res)
            print("x_g is:\n",x_g)
            # x = np.ones(2*m + 2*m**2)
            check_constraints(cons, res.x)
            # sys.exit(10)
            raise ValueError
    else:
        x0 = np.random.rand(2*m + 2*m**2)
        res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
    # print("success",res.success)
    
    return res

def gen_penalty_func(cons):
    def penalty_func(x):
        f = 0
        for con in cons:
            if con['type'] == 'eq':
                # pass
                f = con['fun'](x)**2*PENALTY + f
            elif con['type'] == 'ineq':
                f = min(0, con['fun'](x))**2*PENALTY + f
        return f
    return penalty_func

def gen_feasible_solution(cons, tng, true_initial=False):
    r = len(tng.getGroupList())
    penalty_func = gen_penalty_func(cons)
    if true_initial == True:
        for i  in range(100):
            x0 = np.random.rand(2*r**2)
            res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
            if res.success ==True:
                break
        if i==100:
            print("Iteration for finding initial value exceeds 100 times.")
            sys.exit(10)
    else:
        x0 = np.random.rand(2*r**2)
        res = minimize(penalty_func, x0, method='SLSQP',constraints=cons)
    return res



# Generating jacobi matrix
def gen_jac(tng, conts, mode=0):
    g_list = tng.getGroupList()
    
    r = len(g_list)
    def jac(x):
        grad_f = np.zeros(r)
        grad_t = np.zeros(r)
        # print("x:\n",x)
        x0 = x
        x1 = gen_feasible_solution(conts, tng, True).x
        # check_constraints(conts, x1)
        for v in range(r):
            group = g_list[v]
            com_obj = gen_group_obj(tng, group, mode='com')
            trans_obj = gen_group_obj(tng, group, mode='trans')
            full_obj = gen_group_obj(tng, group)
            
            # print("x0:\n",x0)
            try:
                lam0 = gen_lambda(x0, v, r)
                # cons  = gen_group_constraints(tng, group, x0)
                x_g0 = optimize_group(tng, x0, v).x
                com0 = com_obj(x_g0)
                # print("lam0 %f, com0 %f" % (lam0, com0))
                # print(x_g0)
                # check_constraints(cons, x_g0)
                lam1 = gen_lambda(x1, v, r)
                # cons = gen_group_constraints(tng, group, x1)
                # print("x1:\n",x1)
                x_g1 = optimize_group(tng, x1, v).x
            
                com1 = com_obj(x_g1)
                # print("lam1 %f, com1 %f" % (lam1, com1))
                # print(x_g1)
                # print("lambda:\n", gen_group_lambda(x_g1,2))
                # check_constraints(cons, x_g1)

                grad_f[v] = (com1-com0)/(lam1-lam0)
            except ValueError:
                grad_f[v] = 10

            phi0 = gen_phi(x0,v,r)
            cons = gen_group_constraints(tng, group, x0)
            try:
                x_g0 = optimize_group(tng, x0, v).x
            # print_result(x_g0[2*2:], 2)
            # check_constraints(cons, x_g0)
                trans0 = trans_obj(x_g0)
            # print("phi0 %f, trans0 %f" % (phi0, trans0))

            # x1 = delta_path_ban(x, v, r, -0.5, cons)
                phi1 = gen_phi(x1, v, r)
                # cons = gen_group_constraints(tng, group, x1)
            
                x_g1 = optimize_group(tng, x1, v).x
            # print_result(x_g1, 2)
                trans1 = trans_obj(x_g1)
                grad_t[v] = (trans1 -  trans0)/(phi1 - phi0)
            except ValueError:
                grad_t[v] = 10
            # print("phi1 %f, trans1 %f" % (phi1, trans1)) 
        """ for v in range(r): 
            group = g_list[v]
            com_obj = gen_group_obj(tng, group, mode='com')
            trans_obj = gen_group_obj(tng, group, mode='trans')
            full_obj = gen_group_obj(tng, group)
        
            # Fist， get original
            
            lam0 = gen_lambda(x, v, r)
            cons  = gen_group_constraints(tng, group, x)
            print("x:\n",x)
            x_g0 = optimize_group(tng, x, v).x
            com0 = com_obj(x_g0)
            print("lam0 %f, com0 %f" % (lam0, com0))
            print(x_g0)
            # print("lambda:\n", gen_group_lambda(x_g0,2))
            # check_constraints(cons, x_g0)
            # Then, change
            x1 = delta_offloading(x, v, r, 0.5, cons)

            lam1 = gen_lambda(x1, v, r)
            cons = gen_group_constraints(tng, group, x1)
            print("x1:\n",x1)
            x_g1 = optimize_group(tng, x1, v).x
            
            com1 = com_obj(x_g1)
            print("lam1 %f, com1 %f" % (lam1, com1))
            print(x_g1)
            print("lambda:\n", gen_group_lambda(x_g1,2))
            # check_constraints(cons, x_g1)

            grad_f[v] = (com1-com0)/(lam1-lam0)

            phi0 = gen_phi(x,v,r)
            # cons = gen_group_constraints(tng, group, x)
            x_g0 = optimize_group(tng, x, v).x
            # print_result(x_g0[2*2:], 2)
            # check_constraints(cons, x_g0)
            trans0 = trans_obj(x_g0)
            print("phi0 %f, trans0 %f" % (phi0, trans0))

            x1 = delta_path_ban(x, v, r, 0.5, cons)
            phi1 = gen_phi(x1, v, r)
            # cons = gen_group_constraints(tng, group, x1)
            x_g1 = optimize_group(tng, x1, v).x
            # print_result(x_g1, 2)
            trans1 = trans_obj(x_g1)
            grad_t[v] = (trans1 -  trans0)/(phi1 - phi0)
            print("phi1 %f, trans1 %f" % (phi1, trans1)) """
        print("grad_f:\n",grad_f)
        print("grad_t:\n",grad_t)
        grad_x = np.zeros(len(x))
        if mode==0:
            for u in range(r):
                for v in range(r):
                    grad_x[u*r + v] = grad_f[v]
                    if u != v:
                        grad_x[u*r + v ] = grad_x[u*r + v ] + BETA_D[0]/x[r**2 + u*r + v]
                        grad_x[r**2 + u*r + v] = grad_t[v] + grad_t[u] - x[u*r + v]*BETA_D[0]/(x[r**2 + u*r + v]**2)
        elif mode==1:
            for u in range(r):
                for v in range(r):
                    if u != v:
                        grad_x[u*r + v] = grad_f[v] - grad_f[u] + BETA_D[0]/x[r**2 + u*r + v]
                        grad_x[r**2 + u*r + v] = grad_t[v] + grad_t[u] - x[u*r + v]*BETA_D[0]/(x[r**2 + u*r + v]**2)
        else:
            print("no such mode")
            sys.exit()
        print(grad_x)
        return grad_x
        
    return jac


def gen_ratio(tng, group):
    pass

        # cons = gen_group_constraints(tng, group, x)

def delta_offloading(x, v, r, delta, cons):
    tmp = 0
    tmpx = np.copy(x)
    for u in range(r):
        if u != v:
            tmp = tmp + tmpx[u*r + v]**2
    scale = np.sqrt(tmp)
    ratio = delta/(scale) + 1
    for u in range(r):
        if u != v:
            tmpx[u*r + v] = ratio*tmpx[u*r + v]

    # for c in cons:
    #     if c['fun'](tmpx) < 0:
    #         pass
    # print(np.sum((tmpx-x)**2))
    return tmpx


def delta_path_ban(x, v, r, delta, cons):
    tmp = 0
    tmpx= np.copy(x)
    for u in range(r):
        if u != v:
            tmp = tmp + tmpx[r**2 + u*r + v]**2
            tmp = tmp + tmpx[r**2 + v*r + u]**2
    scale = np.sqrt(tmp)
    ratio = delta/(scale) + 1
    for u in range(r):
        if u != v:
            tmpx[r**2 + u*r + v] = tmpx[r**2 + u*r + v]*ratio
            tmpx[r**2 + v*r + u] = tmpx[r**2 + v*r + u]*ratio
    # print(np.sum((tmpx-x)**2))
    return tmpx

# Some small toools
def gen_group_args(tng, group):
    c = []
    _beta_d = np.array(BETA_D)
    _beta_h = np.array(BETA_H)
    s_list = tng.getServersInGroup(group)
    for s in s_list:
        c.append(s.getRscAmount())
    # print("beta_d:\n",_beta_d)
    # print("beta_h",_beta_h)
    # print("Servers' rsc:\n", c)
    # print("np.dot(beta_d, beta_h)",np.dot(_beta_d, _beta_h))
    return _beta_d, _beta_h, c


        
# Transfrom between inter group and group
def gen_inter_data(x_g, g_index, r):
    gamma_v = 0
    theta_u = 0
    phi_v_in = 0
    phi_v_out = 0
    for u in range(r):
        if u != g_index:
            gamma_v = gamma_v + x_g[u*r + g_index]
            theta_u = theta_u + x_g[g_index*r + u]
            phi_v_in = phi_v_in + x_g[r**2 + u*r + g_index]
            phi_v_out = phi_v_out + x_g[r**2 + g_index*r + u]
    # print("gamma_v, theta_u, phi_v_in, phi_v_out:")
    # print(gamma_v, theta_u, phi_v_in, phi_v_out)
    return gamma_v, theta_u, phi_v_in, phi_v_out

def gen_lambda(x, v, r):
    lam = 0
    # print(x)
    for u in range(r):
        lam = lam + x[u*r + v]
    return lam

def gen_phi(x, v, r):
    phi = 0
    for u in range(r):
        if u != v:
            phi = phi + x[r**2 + u*r + v]
            phi = phi + x[r**2 + v*r + u]
    return phi


# Solver
def corgi(tng, iteration=100, repeat=1):
    #构建问题
    #生成初始解
    #迭代生成最优解
    pass

def optimize_group(tng, x, g_index):
    
    g = tng.getGroupList()[g_index]
    print("in optimize_group, optimized group is %s" % g.getKey())
    # m = len(g.getServerList())
    x = np.around(x, 3)
    min_fun = 100000
    tmp_res =0
    # success = True
    cons = gen_group_constraints(tng, g, x)
    for i in range(20):
        res = gen_group_feasible_solution(cons, tng, g, x, True)
        res = minimize(gen_group_obj(tng, g), res.x, method='SLSQP', constraints=cons)
        if res.success == True and res.fun < min_fun:
            tmp_res =  res
            min_fun = res.fun
            # return res
    if min_fun == 100000:
        print("Can't get a valid solution for %s" % g.getKey())
        # sys.exit()
        tmp_res = gen_group_feasible_solution(cons, tng, g, True)
        # success = False
    else:
        return tmp_res#, success


def test_constraints(tng):
    # cons = gen_constraints(tng)

    # res = gen_feasible_solution(cons, tng, True)
    # check_constraints(cons, res.x)
    # print("res for gen_feasible_solution:")
    # print_result(res.x, len(tng.getGroupList()))

    # g = tng.getGroupList()[0]
    m = 2
    # cons = gen_group_constraints(tng, g, res.x)
    # # x = np.ones((2*m + 2*m**2 ))
    # res = gen_group_feasible_solution(cons, tng, g, True)
    # check_constraints(cons, res.x)
    # x1 = [4.000, 0.000, 0.315, 3.685, 23.589, 2.528, 2.585, 11.058]
    # x1=  [3.977, 0.023, 0.554, 3.446, 23.589, 2.494, 2.397, 11.058]
    x1 = [4.000, 0.000, 1.996, 2.004, 350.970, 3.030, 5.940, 298.740]
    cons = gen_constraints(tng)
    check_constraints(cons, x1)
    g_list = tng.getGroupList()
    for i  in range(len(tng.getGroupList())):
        g = g_list[i]
        # cons = gen_group_constraints(tng, g, x1)
        res = optimize_group(tng, x1, i)
        # res =  gen_group_feasible_solution(cons, tng, g, x1, True)
        print("in %s" % g.getKey())
        print(res)
        print_result(res.x[2*m:],m)
    

def test_group_obj(tng):
    cons = gen_constraints(tng)

    res = gen_feasible_solution(cons, tng, True)
    check_constraints(cons, res.x)
    print("res for gen_feasible_solution:")
    print_result(res.x, len(tng.getGroupList()))

    g = tng.getGroupList()[0]
    m = len(g.getServerList())
    cons = gen_group_constraints(tng, g, res.x)
    # x = np.ones((2*m + 2*m**2 ))
    res = gen_group_feasible_solution(cons, tng, g, True)
    for i in range(3):
        res = minimize(gen_group_obj(tng, g), res.x, method='SLSQP', constraints=cons)
        print(res.success)
        print(res.fun)
        print_result(res.x, len(tng.getGroupList()))
    check_constraints(cons, res.x)

def test_obj(tng):
    cons = gen_constraints(tng)

    x0 = gen_feasible_solution(cons, tng, True).x

    trace = []
    def callback(xk):
        trace.append(xk)

    for i in range(50):
        
        x0 = gen_feasible_solution(cons, tng, True).x
        # res = minimize(gen_obj(tng), x0, method='SLSQP', callback=callback, constraints=cons, options={'maxiter':30, 'eps':0.05}, jac='2-point')#, jac=gen_jac(tng,cons, mode=0)) # Good result
        # x0 = gen_feasible_solution(cons, tng, True).x
        res = minimize(gen_obj(tng), x0, method='SLSQP', callback=callback, constraints=cons, options={'maxiter':30, 'eps':0.05}, jac=gen_jac(tng,cons, mode=1)) # Good result
        # x0 = res.x
        print(res)
    
    print("history")
    obj = gen_obj(tng)
    for e in trace:
        print(obj(e))

def test_deltas(tng):
    cons = gen_constraints(tng)

    res = gen_feasible_solution(cons, tng, True)
    check_constraints(cons, res.x)
    print("res for gen_feasible_solution:")
    print_result(res.x, len(tng.getGroupList()))

    x0 = res.x
    print("x1:\n",x0)
    print("after applying delta_offloading:\n", delta_offloading(x0, 0, 2, 0.5, cons))
    print("after applying delta_path_ban:\n", delta_path_ban(x0, 0, 2, 0.5, cons))

def test_jac(tng):
    
    cons = gen_constraints(tng)
    jac = gen_jac(tng,cons, mode=1)

    x = [3.976, 0.024, 0.534, 3.466, 23.589, 2.484, 2.380, 11.058]
    grad_x = jac(x)
    print(grad_x)

def test_lam_phi(tng):
    cons = gen_constraints(tng)
    x = gen_feasible_solution(cons, tng, True).x
    print("x:\n",x)
    print("lambda:", gen_lambda(x, 0, 2))
    print("phi:", gen_phi(x, 0, 2))


if __name__ == "__main__":
    np.random.seed(7)
    tng = createATreeGraph()
    tng.print_info()
    # np.random.seed(1)
    np.set_printoptions(formatter={'float':'{:.3f}'.format})

    test_obj(tng)

    # test_jac(tng)
    # test_lam_phi(tng)
    # test_deltas(tng)
    # test_group_obj(tng)
    # test_constraints(tng)


