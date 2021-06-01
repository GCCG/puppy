from ..net_graph import NetGraph, createATreeGraph, TreeNetGraph
from ..net_ap import NetAP
from ..net_link import NetLink
from .. import parameters
import numpy as np
from scipy.optimize import minimize
import sys
from matplotlib import pyplot as plt
import json
from .wolf import check_constraints
import math
import time



TYPE_GROUP = 'group'
TYPE_SERVER = 'server'
TYPE_DUMMY = 'dummy'
INFINITY = 10000000
PENALTY = 1000000
BETA_D = [1,0.2,0.1]
SIGMA_D = [1,1,1]
BETA_H = [0.5,2,0.3]
SIGMA_H = [1,1,1]
TASK_TYPE_LIST = [parameters.CODE_TASK_TYPE_VR, parameters.CODE_TASK_TYPE_VA, parameters.CODE_TASK_TYPE_IoT]

ADJUST = 0.0001

# Only for two-layer tng
def transform_tng(tng):
    """ 将tng转化成一棵以列表形式表达的树 """
    root = create_node()
    root['type'] = TYPE_GROUP

    # 设置子孙
    children = []
    g_list = tng.getGroupList()
    bl_list = tng.get_backhaul_links()
    for g in g_list:
        s_list = tng.getServersInGroup(g) 
        switch = tng.getSwitchInGroup(g)
        g_tmp = create_node()
        g_tmp['type'] = TYPE_GROUP
        # 设置域的输入、输出带宽
        for l in bl_list:
            print("backhaul link ban:",l.getBandwidth())
            if l.getHeadAP().getKey() == switch.getKey():
                g_tmp['outlink_ban'] = l.getBandwidth()
            elif l.getTailAP().getKey() == switch.getKey():
                g_tmp['inlink_ban'] = l.getBandwidth()

        # 设置域的子孙
        l_list = tng.getLinksInGroup(g)
        g_children = []
        for s in s_list:
            s_tmp = create_node()
            s_tmp['type'] = TYPE_SERVER
            # 设置服务器的计算资源量
            s_tmp['rsc'] = s.getRscAmount()
            s_tmp['alpha'] = s.getGroup().getTaskGenInfo(TASK_TYPE_LIST[0])[0]
            # 设置服务器的输入、输出带宽
            for l in l_list:
                if l.getHeadAP().getKey() == s.getKey():
                    s_tmp['outlink_ban'] = l.getBandwidth()
                elif l.getTailAP().getKey() == s.getKey():
                    s_tmp['inlink_ban'] = l.getBandwidth()
            s_tmp['total_ban'] = 0
            s_tmp['leaf_num'] = 1
            g_children.append(s_tmp)
        g_tmp['children'] = g_children

        # 设置域的计算资源量
        g_rsc = 0
        g_alpha = 0
        g_total_ban = 0
        g_leaf_num = 0
        for c in g_children:
            g_rsc = g_rsc + c['rsc']
            g_alpha = g_alpha + c['alpha']
            g_total_ban = g_total_ban + c['inlink_ban'] + c['outlink_ban']
            g_leaf_num = g_leaf_num + c['leaf_num']
        g_tmp['rsc'] = g_rsc
        g_tmp['alpha'] = g_alpha
        g_tmp['total_ban'] = g_total_ban
        children.append(g_tmp)
    root['children'] = children

    # 设置资源量
    r_rsc = 0
    r_alpha = 0
    r_total_ban = 0
    for c in children:
        r_rsc = r_rsc + c['rsc']
        r_alpha = r_alpha + c['alpha']
        r_total_ban = c['inlink_ban'] + c['outlink_ban'] + r_total_ban
    root['rsc'] = r_rsc
    root['alpha'] = r_alpha
    root['total_ban'] = r_total_ban

    return root

def reconstruct_tree(tree, degree):
    """ 将问题化为一棵问题树
    重构时是以后序的方式遍历tng的；
    输入是tng
    返回是一个以列表形式表达的树 """
    # 首先将子结点全部转换掉
    children = tree['children']
    for i in range(len(children)):
        children[i] = reconstruct_tree(children[i], degree)
    # 然后转换自身
    if len(children) > 0 :
        # tree['type'] = TYPE_DUMMY
        while len(children) > degree:
            tmp = []
            # 等量分割并生成结点
            for i in range(int(len(children)/degree)):
                node = create_node()
                node['type'] = TYPE_DUMMY
                node['children'] = children[i*degree:(i+1)*degree]
                tmp.append(node)
            # 将剩余的放在一个结点中
            if len(children)%degree>0:
                node = create_node()
                node['type'] = TYPE_DUMMY
                node['children'] = children[len(children)-len(children)%degree:len(children)]
                tmp.append(node)
            children = tmp
    tree['children'] = children
    return tree

def create_node():
    tmp = {}
    # 任务生成速率
    tmp['alpha'] = None
    # 计算资源总量
    tmp['rsc'] = None
    # 结点类型
    tmp['type'] = None
    # 父结点指向自身的连接带宽
    tmp['inlink_ban'] = None
    # 自身指向父结点的连接带宽
    tmp['outlink_ban'] = None
    # 结点外任务输入速度
    tmp['input_load'] = None
    # 对外任务输出速度
    tmp['output_load'] = None
    # 父结点对自身输入带宽
    tmp['input_ban'] = None
    # 自身对父结点输出带宽
    tmp['output_ban'] = None
    # 自身的子结点
    tmp['children'] = []
    # 自身的约束
    tmp['constraints'] = []
    # 子结点连接自身的所有连接的带宽和
    tmp['total_ban'] = None
    # 结点名称
    tmp['name'] = 'Noname'
    # 问题的解
    tmp['solution'] = None
    # 承载的任务量
    tmp['lambda'] = None
    # 叶结点数目
    tmp['leaf_num'] = None
    # 子域间任务的传输时间
    tmp['trans_time'] = 0
    # 任务的计算时间
    tmp['com_time'] = 0
    
    return tmp


def construct_sub_problems(tree, name):
    """ 以后序遍历的顺序构造各个问题；
    每个问题的信息以字典的形式存储；
    输入、输出都是树，"""
    n = 0
    for c in tree['children']:
        construct_sub_problems(c, name +'-' + str(n))
        n = n + 1
    if tree['rsc'] is None and tree['type'] == TYPE_DUMMY:
        rsc = 0
        for c in tree['children']:
            rsc = rsc + c['rsc']
        tree['rsc'] = rsc
    if tree['inlink_ban'] is None and tree['type'] == TYPE_DUMMY:
        in_link_ban = 0
        for c in tree['children']:
            in_link_ban = in_link_ban + c['inlink_ban']
        tree['inlink_ban'] = in_link_ban/min(2, len(tree['children']))
    if tree['outlink_ban'] is None and tree['type'] == TYPE_DUMMY:
        out_link_ban = 0
        for c in tree['children']:
            out_link_ban = out_link_ban + c['outlink_ban']
        tree['outlink_ban'] = out_link_ban//min(2, len(tree['children']))
    if tree['alpha'] is None and tree['type'] == TYPE_DUMMY:
        alpha = 0
        for c in tree['children']:
            alpha = alpha + c['alpha']
        tree['alpha'] = alpha
    if tree['total_ban'] is None and tree['type'] == TYPE_DUMMY:
        total_ban = 0
        for c in tree['children']:
            total_ban = total_ban + c['inlink_ban'] + c['outlink_ban']
        tree['total_ban'] = total_ban
    tree['name'] = name
    leaf_num = 0
    for c in tree['children']:
        leaf_num = leaf_num + c['leaf_num']
    if leaf_num > 0:
        tree['leaf_num'] = leaf_num
    
def traverse_tree(tree, name):
    if tree['name'] == name:
        return tree
    else:
        for child in tree['children']:
            node = traverse_tree(child, name)
            if node != None and  node['name'] == name:
                return node

def load_tree(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def store_tree(tree, file_name):
    with open(file_name, 'w') as f:
        json.dump(tree, f, sort_keys=True, indent=4, separators=(',', ': '))
        # f.close()


#####------------hierarchy求解器-------------############
def solve_problems(tree, max_repeat=30, max_success=5):
    """ 以先序遍历的顺序求解子问题
    问题的解放在字典里
    输入、输出都是树 """
    # 填充约束
    cons = gen_cons(tree, BETA_D[0])
    # 解决当前问题
    children = tree['children']
    iteration = 0
    min_fun = 10000000000000
    min_res = None
    success_count = 0
    if len(children[0]['children']) > 0: # 如果存在孙子
        obj = gen_obj_for_nongroup(tree, mode=3)
        while iteration < max_repeat and success_count < max_success:
            print("iteration ",iteration)
            try:
                x0 = gen_feasible_solution(cons, tree).x
            except ValueError:
                return False
            res = minimize(obj, x0, method='SLSQP', constraints=cons)
            iteration = iteration + 1
            # print(res.fun)
            # print(res.message)
            if res.success == True and res.fun > 0:
                success_count = success_count + 1
                # print("success_count: ",success_count)
                # print(res.fun)
            # print(res.success)
            if res.fun < min_fun and res.fun > 0 and res.success == True:
                min_fun = res.fun
                min_res = res
    elif children[0]['type'] == TYPE_SERVER: # 如果不存在孙子
        obj = gen_obj_for_group(tree)
        while iteration < max_repeat and success_count < max_success:
            print("iteration ",iteration)
            try:
                x0 = gen_feasible_solution(cons, tree).x
            except ValueError:
                return False
            res = minimize(obj, x0, method='SLSQP', constraints=cons)
            # print(res.fun)
            # print(res.message)
            iteration = iteration + 1
            # print(res.fun)
            # print(res.success)
            if res.success == True and res.fun > 0:
                success_count = success_count + 1
                # print("success_count: ",success_count)
                # print(res.fun)
            if res.fun < min_fun and res.fun > 0 and res.success == True:
                # tree['solution'] = result_to_dict(res.x, len(children))
                min_fun = res.fun
                min_res = res
    else:
        print("Something is wrong with your tree.")
        sys.exit()
    if min_res == None:
        return False 
    tree['solution'] = result_to_dict(min_res.x, len(children))
    print("In %s, type is %s, res is:" % (tree['name'], tree['type']))
    print(min_res)

    # 给子问题设置参数
    x = min_res.x
    r = len(children)
    tree['trans_time'] = trans_time(x, r, beta_d=BETA_D[0])
    gamma_k = tree['input_load']
    theta_k = tree['output_load']
    phi_k_in = tree['input_ban']
    phi_k_out = tree['output_ban']

    phi_v_in = phi_v_ins(x, gamma_k, phi_k_in, r)
    phi_v_out = phi_v_outs(x, theta_k, phi_k_out, r)
    gamma_v = gamma_vs(x, r)
    theta_v = theta_vs(x, r)
    lam_v = lam_vs(x, r)
    mu_v = []
    ban_list = []
    for v in range(r):
        mu_v.append(children[v]['rsc']/BETA_H[0])
        ban_list.append(children[v]['inlink_ban'])
    print("trans_time:",trans_time(x, r, BETA_D[0]))
    print("com_time:", com_time(x, r, BETA_H[0], SIGMA_H[0], tree))
    print("inlink_bans:", ban_list)
    print("gamma_v:",gamma_v)
    print("theta_v:",theta_v)
    print("phi_v_in:",phi_v_in)
    print("phi_v_out:",phi_v_out)
    print("lams:",lam_v)
    print("mu_v:", mu_v)
    omega = np.array(mu_v)/lam_v
    print("omega:",omega)
    k = 0
    psi = cal_psi(SIGMA_H[0], BETA_H[0])
    m = 0
    for child in children:
        print("leaf_num: ",child['leaf_num'])
        print("m_v*h(omega_v): %f" % (child['leaf_num']*h(omega[k], psi)))
        k = k + 1
        if len(child['children']) > 0:
            m = m + len(child['children'])
        else:
            m = m + 1
    omega_0 = sum(mu_v)/sum(lam_v)
    print("omega_0:",omega_0)
    print("leaf_num: ", tree['leaf_num'])
    print("m*h(omega_0)", tree['leaf_num']*h(omega_0, psi))
    
    # check_constraints(cons, x)
    for v in range(r):
        children[v]['input_load'] = gamma_v[v]
        children[v]['output_load'] = theta_v[v]
        children[v]['input_ban'] = phi_v_in[v]
        children[v]['output_ban'] = phi_v_out[v]
        children[v]['lambda'] = lam_v[v]

    # 递归解决子问题
    if len(children[0]['children']) > 0:
        for child in children:
            success = solve_problems(child, max_repeat, max_success)
            if success == False:
                return False
    else:
        tree['com_time'] = com_time(x=x, r=r, beta_h=BETA_H[0], sigma_h=SIGMA_H[0], node=tree)
    return True

def cal_total_time_from_tree(tree):
    time = tree['com_time'] + tree['trans_time']
    for child in tree['children']:
        time = time + cal_total_time_from_tree(child)
    return time

def cal_total_com_time(tree):
    time = 0
    if tree['type'] == 'server':
        time = f_v(tree['lambda'], mu_v=tree['rsc']/BETA_H[0], beta_h=BETA_H[0], sigma_h=SIGMA_H[0])
    for child in tree['children']:
        time = time + cal_total_com_time(child)
    return time


# 从解中计算传输时间
def trans_time(x, r, beta_d):
    f = 0
    for v in range(r):
        for u in range(r):
            if u != v and x[2*r + r**2 + u*r + v] > 0:
                f =  f + x[2*r + u*r + v]*beta_d/x[2*r + r**2 + u*r + v]
    return f

def com_time(x, r, beta_h, sigma_h, node):
    # rsc_list = []
    # for child in node['children']:
    #     rsc_list.append(child['rsc'])
    # rsc_list = np.array(rsc_list)
    # mu = rsc_list/beta_h
    # lam_vs = np.zeros(r)
    # for v in range(r):
    #     lam_vs[v] = x[v]
    #     for u in range(r):
    #         lam_vs[v] = lam_vs[v] + x[2*r + u*r + v]
    # f = 0
    if len(node['children'][0]['children']) > 0:
        # print("nongroup")
        obj = gen_obj_for_nongroup(node, need_trans_time=False, mode=3)
    else:
        # print("group")
        obj = gen_obj_for_group(node, need_trans_time=False)
    # for v in range(r):
    #     f = f + f_v(lam_vs[v], mu[v], sigma_h, beta_h)
   
    return obj(x)
    # return f

# 计算
def phi_v_ins(x, gamma_k, phi_k_in, r):
    ins = np.zeros(r)
    for v in range(r):
        if gamma_k > 0:
            ins[v] = x[v]*phi_k_in/gamma_k
        for u in range(r):
            if u != v:
                ins[v] = ins[v] + x[2*r + r**2 + u*r + v]
    return ins

def phi_v_outs(x, theta_k, phi_k_out, r):
    outs = np.zeros(r)
    for v in range(r):
        if theta_k > 0:
            outs[v] = x[r+v]*phi_k_out/theta_k
        for u in range(r):
            if u != v:
                outs[v] = outs[v] + x[2*r + r**2 + v*r + u]
    return outs

def gamma_vs(x, r):
    gams = np.zeros(r)
    for v in range(r):
        gams[v] = x[v]
        for u in range(r):
            if u != v:
                gams[v] = gams[v] + x[2*r + u*r + v]
    return gams

def theta_vs(x, r):
    thes = np.zeros(r)
    for v in range(r):
        thes[v] = x[r + v]
        for u in range(r):
            if u != v:
                thes[v] = thes[v] + x[2*r + v*r + u]
    return thes

def lam_vs(x, r):
    lams = np.zeros(r)
    for v in range(r):
        lams[v] = x[v]
        for u in range(r):
            lams[v] = lams[v] + x[2*r + u*r + v]
    return lams

def result_to_dict(x, r):
    result = {}
    result['output_decisions'] = x[r:2*r].tolist()
    result['input_decisions'] = x[0:r].tolist()
    result['inner_offloading'] = np.reshape(x[2*r:2*r + r**2], (r,r)).tolist()
    result['inner_path_ban'] = np.reshape(x[2*r + r**2: 2*r + r**2*2], (r,r)).tolist()
    return result


# 用于生成可行解的函数
def gen_feasible_solution(cons, node, true_initial=True, gen_feasible='linear'):
    if gen_feasible == 'penalty':
        obj = gen_penalty_func(cons)
    elif gen_feasible == 'linear':
        obj = linear_obj
    # penalty_func = gen_penalty_func(cons)
    r = len(node['children'])
    if true_initial == True:
        for i  in range(100):
            x0 = np.random.rand(2*r + 2*r**2)*2
            res = minimize(obj, x0, method='SLSQP',constraints=cons)
            if res.success ==True:
                break
        if i==99:
            print("-------------------Failed----------------------")
            print(res)
            # x0 = np.zeros((2*r + 2*r**2))
            check_constraints(cons, x0)
            x0 = np.zeros((2*r + 2*r**2))
            check_constraints(cons, x0)
            print("Iteration for finding initial value exceeds 100 times.")
            raise ValueError("Can't find feasible solution.")
    # print(res.success)
    # if node['name'] == 'tng-0-0-0':
    #     check_constraints(cons, x0)
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

def linear_obj(x):
    return sum(x)


# 用于生成约束的函数
def gen_cons(node, beta_d):
    gamma_k = node['input_load']
    theta_k = node['output_load']
    phi_k_in = node['input_ban']
    phi_k_out = node['output_ban']
    children = node['children']
    r = len(children)
    if r == 0:
        print("This node doesn't represent group.")
        sys.exit()
    
    alphas = np.zeros(r)
    mus = np.zeros(r)
    for v in range(r):
        mus[v] = children[v]['rsc']/BETA_H[0]
        alphas[v] = children[v]['alpha']

    length = 2*r + r**2*2
    # 针对域输入负载的约束
    # 针对域输出负载的约束
    C1 = np.zeros((2,length))
    b1 = np.zeros(2)
    type1 = ['eq', 'eq']
    info1 = ['remaining input of %s' % node['name'], 'remaining output of %s'  % node['name']]
    for j in range(r):
        C1[0, j] = -1
        C1[1, r + j] = -1 
    b1[0] = gamma_k
    b1[1] = theta_k
    b1[0] = max(0, gamma_k) # 注意，注意,gamma_k和theta_K一定要保证大于0，否则问题经常会出现无解的情况。
    b1[1] = max(0, theta_k)

    # 针对子域输出负载的约束
    C2 = np.zeros((r, length))
    b2 = np.zeros(r)
    type2 = []
    info2 = []
    for i in range(r):
        C2[i, r + i] = -1
        b2[i] = alphas[i]
        type2.append('eq')
        info2.append("remaning load of %s" % children[i]['name'])
        for j in range(r):
            C2[i, 2*r + i*r + j] = -1 
    # 针对子域输入负载的约束
    C3 = np.zeros((r, length))
    b3 = np.zeros(r)
    type3 = []
    info3 = []
    for j in range(r):
        C3[j, j] = -1
        b3[j] = mus[j]
        type3.append('ineq')
        info3.append("remaining rsc of %s" % children[j]['name'])
        for i in range(r):
            C3[j, 2*r + i*r + j] = -1
    # 针对路径带宽的约束
    C4 = np.zeros((r**2, length))
    b4 = np.zeros(r**2)
    type4 = []
    info4 = []
    for i in range(r):
        for j in range(r):
            type4.append('ineq')
            info4.append("surplus bandwidth of path from %s to %s" % (children[i]['name'], children[j]['name']))
            C4[i*r + j, 2*r + r**2 + i*r + j] = 1
            C4[i*r + j, 2*r + i*r + j] = -beta_d
    # 针对连接带宽的约束
    l_num = 2*r #域内共2r条连接，每个子域出入共两条。
    C5 = np.zeros((l_num, length))
    b5 = np.zeros(l_num)
    type5 = []
    info5 = []
    i = 0 
    for v in range(len(children)):
        # 先约束输出
        child = children[v]
        b5[i] = child['outlink_ban']
        type5.append('ineq')
        info5.append("remaining bandwidth of %s of %s" % ('outlink', child['name']))
        # for u in range(r):
        if theta_k > 0.001:
            C5[i, r + v] = - phi_k_out/theta_k
        for u in range(r):
            if u != v:
                C5[i, 2*r + r**2 + v*r + u] = -1
        i = i + 1
        # 再约束输入
        b5[i] = child['inlink_ban']
        type5.append('ineq')
        info5.append("remaining bandwidth of %s of %s" % ('inlink', child['name']))
        if gamma_k > 0.001:
            C5[i, v] = - phi_k_in/gamma_k
        for u in range(r):
            if u != v:
                C5[i, 2*r + r**2 + u*r + v] = -1 
        i = i + 1
            

    # 针对子域在本域内输入、输出卸载量的约束
    C6 = np.zeros((r**2, length)) 
    b6 = np.zeros(r**2) # 这里不能减去ADJUST，会导致求解效率变差，解的效果变差。
    type6 = []
    info6 = []
    for i in range(r):
        for j in range(r):
            type6.append('ineq')
            info6.append("offloading from %s to %s" % (children[i]['name'], children[j]['name']))
            C6[i*r + j, 2*r + i*r + j] = 1
    # 针对子域在本域外输入、输出卸载量的约束
    C7 = np.zeros((2*r, length))
    b7 = np.zeros(2*r)
    type7 = []
    info7 = []
    for i in range(r):
        type7.append('ineq')
        info7.append("input decision of %s" % children[i]['name'])
        C7[i, i] = 1
        type7.append('ineq')
        info7.append("output decision of %s" % children[j]['name'])
        C7[r + i, r + i] = 1
    


    # 合并所有约束
    C = np.concatenate((C1, C2, C3, C4, C5, C6, C7), axis=0)
    b = np.concatenate((b1, b2, b3, b4, b5, b6, b7))
    info = info1 + info2 + info3 + info4 + info5 + info6 + info7
    _type = type1 + type2 + type3 + type4 + type5 + type6 + type7

    # 将约束转化成求解器需要的形式
    cons = []
    for i in range(len(b)):
        tmp = {}
        tmp['type'] = _type[i]
        tmp['info'] = info[i]
        tmp['fun'] = lambda x, i=i: np.dot(x, C[i]) + b[i]
        cons.append(tmp)
    
    # 针对子域带宽的约束 ---这是个子域内带宽资源的下界约束，同时也是子域内传输时间的上界约束。
    if len(children[0]['children']) > 0: # 如果子域并非叶子的话，才会设置下面的约束
        for v in range(r):
            alphas_v = []
            rsc_list_v = []
            in_ban_list_v = []
            out_ban_list_v = []
            child = children[v]
            for gs in child['children']:
                alphas_v.append(gs['alpha'])
                rsc_list_v.append(gs['rsc'])
                in_ban_list_v.append(gs['inlink_ban'])
                out_ban_list_v.append(gs['outlink_ban']) 
            alphas_v = np.array(alphas_v)
            rsc_list_v = np.array(rsc_list_v)
            in_ban_list_v = np.array(in_ban_list_v)
            out_ban_list_v = np.array(out_ban_list_v)
            # print("ban sum: ",sum(in_ban_list_v))
            in_fun = lambda x, v=v, alphas_v=alphas_v, rsc_list_v=rsc_list_v, in_ban_list_v=in_ban_list_v:\
                subgroup_ban_constraint_2(x, r, v, alphas_v, rsc_list_v, in_ban_list_v, beta_d, phi_k_in, phi_k_out, gamma_k, theta_k, direction='in')
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = in_fun
            tmp['info'] = 'lower bound of remainding in-link bandwith in subgroup %s' % child['name']
            cons.append(tmp)

            out_fun = lambda x, v=v, alphas_v=alphas_v, rsc_list_v=rsc_list_v, out_ban_list_v=out_ban_list_v:\
                subgroup_ban_constraint(x, r, v, alphas_v, rsc_list_v, out_ban_list_v, beta_d, phi_k_in, phi_k_out, gamma_k, theta_k, direction='out')
            tmp = {}
            tmp['type'] = 'ineq'
            tmp['fun'] = out_fun
            tmp['info'] = 'lower bound of remainding out-link bandwith in subgroup %s' % child['name']
            cons.append(tmp)
    
    return cons

def subgroup_ban_constraint(x, r, v, alphas_v, rsc_list_v, ban_list_v, beta_d, phi_in_k, phi_out_k, gamma_k, theta_k, direction):
    lam_v = x[v]
    if gamma_k > 0:
        phi_in_v = x[v]*phi_in_k/gamma_k
    else:
        phi_in_v = 0
    if theta_k > 0:
        phi_out_v = x[r + v]*phi_out_k/theta_k
    else:
        phi_out_v = 0
    gamma_v = 0
    theta_v = 0
    for u in range(r):
        lam_v = lam_v + x[2*r + u*r + v]
        if u != v:
            gamma_v = gamma_v + x[2*r + u*r + v]
            theta_v = theta_v + x[2*r + r**2 + v*r + u]
            phi_in_v = phi_in_v + x[2*r + r**2 + u*r + v]
            phi_out_v = phi_out_v + x[2*r + r**2 + v*r + u]
    if direction =='in':
        # print("inner trans: ", gen_inner_trans(alphas_v, lam_v, rsc_list_v))
        # if gamma_v > 0:
        #     return sum(ban_list_v) - phi_in_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'in')
        # else:
        #     return sum(ban_list_v) - phi_in_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'in')
        return sum(ban_list_v) - phi_in_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'in')
        # return gamma_v*sum(ban_list_v)/2 - lam_v*phi_in_v
    elif direction == 'out':
        # if theta_v > 0:
        #     return sum(ban_list_v) - phi_out_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v)
        # else:
        #     return sum(ban_list_v) - phi_out_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v)
        return sum(ban_list_v) - phi_out_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'out')
        
        # return theta_v*sum(ban_list_v)/2 - lam_v*phi_out_v
        # return max(0,3*theta_v) -phi_out_v
    else:
        raise ValueError("direction value is not valid.")

def gen_inner_trans(alphas, lam_v, rsc_list, direction):
    lam = np.zeros(len(rsc_list))
    c_v = sum(rsc_list)
    for j in range(len(rsc_list)):
        lam[j] = lam_v*rsc_list[j]/c_v
    trans_amount = 0
    if direction == 'in':
        for j in range(len(rsc_list)):
            # trans_amount = trans_amount + abs(alphas[j]-lam[j])
            trans_amount = trans_amount + max(0, lam[j] - alphas[j])
    elif direction == 'out':
        for j in range(len(rsc_list)):
            trans_amount = trans_amount + max(0, alphas[j] - lam[j])
    else:
        raise ValueError("direction value is not valied.")
    return trans_amount

def subgroup_ban_constraint_2(x, r, v, alphas_v, rsc_list_v, ban_list_v, beta_d, phi_in_k, phi_out_k, gamma_k, theta_k, direction):
    lam_v = x[v]
    if gamma_k > 0:
        phi_in_v = x[v]*phi_in_k/gamma_k
    else:
        phi_in_v = 0
    if theta_k > 0:
        phi_out_v = x[r + v]*phi_out_k/theta_k
    else:
        phi_out_v = 0
    gamma_v = 0
    theta_v = 0
    for u in range(r):
        lam_v = lam_v + x[2*r + u*r + v]
        if u != v:
            gamma_v = gamma_v + x[2*r + u*r + v]
            theta_v = theta_v + x[2*r + r**2 + v*r + u]
            phi_in_v = phi_in_v + x[2*r + r**2 + u*r + v]
            phi_out_v = phi_out_v + x[2*r + r**2 + v*r + u]
    if direction =='in':
        # return sum(ban_list_v) - phi_in_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'in')
        return sum(ban_list_v) - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'in')*sum(ban_list_v)/sum(rsc_list_v) - phi_in_v 
    elif direction == 'out':
        # return sum(ban_list_v) - phi_out_v - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'out')
        return sum(ban_list_v) - beta_d*gen_inner_trans(alphas_v, lam_v, rsc_list_v, 'out')*sum(ban_list_v)/sum(rsc_list_v) - phi_out_v
    else:
        raise ValueError("direction value is not valid.")


# 生成目标函数
def gen_obj_for_nongroup(node, mode=0, need_trans_time=True):
    """ 对于非叶结点，优化函数是上界 """
    gamma_k = node['input_load']
    phi_k_in = node['input_ban'] 
    phi_k_out = node['output_ban']
    theta_k = node['output_load']
    beta_d = BETA_D[0]
    beta_h = BETA_H[0]
    r = len(node['children']) # 域内节点数目
    rsc_list = []
    ban_list = []
    alpha_list = []
    for child in node['children']:
        rscs = []
        bans = []
        alphas = []
        if len(child['children']) < 0:
            print("Invalid node, it has no grandson")
            sys.exit()
        for gs in child['children']:
            rscs.append(gs['rsc'])
            bans.append(gs['inlink_ban'] + gs['outlink_ban'])
            alphas.append(gs['alpha'])
        
        rsc_list.append(np.array(rscs))
        ban_list.append(np.array(bans))
        alpha_list.append(np.array(alphas))
    # print("ban_list:\n",ban_list)
    psi  = cal_psi(SIGMA_H[0], BETA_H[0])
    def obj(x):
        lam_vs = np.zeros(r)
        phi_vs = np.zeros(r)
        theta_vs = np.zeros(r)
        for v in range(r):
            lam_vs[v] = x[v]
            theta_vs[v] = x[r + v]
            if gamma_k > 0:
                 phi_vs[v] = x[v]*phi_k_in/gamma_k
            if theta_k > 0:
                phi_vs[v] = phi_vs[v] + x[r + v]*phi_k_out/theta_k
            for u in range(r):
                lam_vs[v] = lam_vs[v] + x[2*r + u*r + v]
                if u !=v:
                    theta_vs[v] = theta_vs[v] + x[2*r + v*r + u]
                    phi_vs[v] = phi_vs[v] + x[2*r + r**2 + u*r + v] + x[2*r + r**2 + v*r + u]
        # print("lam_vs:\n", lam_vs)
        # print("rsc_list:\n", rsc_list)
        lam = gen_lam(x, r)
        f = 0
        for v in range(r):
            # print("leaf_num of %s is %d." % ( node['children'][v]['name'], node['children'][v]['leaf_num']))
            f =  f + upper_bound(alpha_list[v], rsc_list[v], ban_list[v], psi, lam_vs[v], phi_vs[v], theta_vs[v], beta_d, beta_h, node['children'][v], mode)
            # print("upperbound for %d" % v)
            # print(upper_bound(alpha_list[v], rsc_list[v], ban_list[v], psi, lam_vs[v], phi_vs[v], theta_vs[v], beta_d, beta_h, node['children'][v], mode))
            if need_trans_time == True:
                for u in range(r):
                    if u != v and x[2*r + r**2 + u*r + v] > 0:
                        f =  f + x[2*r + u*r + v]*beta_d/x[2*r + r**2 + u*r + v] #+ 0.001 * x[2*r + r**2 + u*r + v]
        return f
    return obj

def gen_obj_for_group(node, need_trans_time=True):
    gamma_k = node['input_load']
    phi_k_in = node['input_ban'] 
    phi_k_out = node['output_ban']
    theta_k = node['output_load']
    sigma_h = SIGMA_H[0]
    beta_h = BETA_H[0]
    beta_d = BETA_D[0]
    r = len(node['children']) # 域内节点数目
    rsc_list = []
    for child in node['children']:
        rsc_list.append(child['rsc'])
    rsc_list = np.array(rsc_list)
    mu = rsc_list/beta_h
    def obj(x):
        # 设置各个服务器的负载lam_v
        lam_vs = np.zeros(r)
        for v in range(r):
            lam_vs[v] = x[v]
            for u in range(r):
                lam_vs[v] = lam_vs[v] + x[2*r + u*r + v]
        f = 0
        # print("lam_vs: ",lam_vs)
        # print("mu: ",mu)
        # print("rsc_list:", rsc_list)
        for v in range(r):
            f = f + f_v(lam_vs[v], mu[v], sigma_h, beta_h)
            if need_trans_time:
                for u in range(r):
                    if u != v and x[2*r + r**2 + u*r + v] > 0:
                        f = f + x[2*r + u*r + v]*beta_d/x[2*r + r**2 + u*r + v]
        return f
    return obj

           
def f_v(lam_v, mu_v, sigma_h, beta_h):
    """ 当被优化的结点的子结点已经是边缘服务器时，采用的时间估计函数 """
    return (1/mu_v + (sigma_h**2 + beta_h**2)*lam_v / (2*mu_v * beta_h**2 * (mu_v-lam_v)))*lam_v

def f_v2(lam_v, mu_v, sigma_h, beta_h):
    return (1/mu_v + ((sigma_h/(mu_v*beta_h))**2*mu_v**2+1)/(2*mu_v)*lam_v/(mu_v-lam_v))*lam_v


def gen_lam(x, r):
    """ 计算域内各个节点的任务负载 """
    lam = np.zeros(r)
    for v in range(r):
        lam[v] = x[v]
        for u in range(r):
            lam[v] = x[2*r + u*r + v]
    return lam

def gen_mu(rsc_list):
    return rsc_list/BETA_H[0]

def upper_bound(alpha, rscs, link_bans, psi, lam_v, phi_v, theta_v, beta_d, beta_h, node, mode=0):
    lam = np.zeros(len(rscs))
    c_v = np.sum(rscs)
    mu = rscs/beta_h
    for j in range(len(lam)):
        lam[j] = rscs[j]*lam_v/c_v
    clb = None
    if mode==0:
        clb = com_upper_bound_1
    elif mode==1:
        clb = com_upper_bound_2
    elif mode==2:
        clb = com_upper_bound_3
    elif mode==3:
        clb = com_upper_bound_4
    return clb(psi, mu, lam_v, node) + trans_upper_bound_2(alpha, phi_v, theta_v, beta_d, link_bans, lam)


# d代表任务量，也就是建模当中的lambda
def com_upper_bound_1(psi, mu, lam_v, node):
    """ 只考虑最简单的情况 """
    omega = sum(mu)/lam_v
    # return len(mu) * h(omega, psi) 
    # print("leaf_num of %s is %d" % (node['name'] ,node['leaf_num']))
    return node['leaf_num'] * h(omega, psi)

def com_upper_bound_2(psi, mu, lam_v, node):
    omega = sum(mu)/lam_v
    # return min(len(mu)*h(omega, psi), H_r(mu, len(mu), lam_v, psi))
    return min(len(mu)*h(omega, psi), H_r(mu, len(mu), lam_v, psi))

def com_upper_bound_3(psi, mu, lam_v, node):
    """ 综合考虑 """
    H_list = []
    for i in range(len(mu)):
        H_list.append(H_r(mu, i, lam_v, psi))
    return min(H_list)

def com_upper_bound_4(psi, mu, lam_v, node):
    avg_mu = sum(mu)/len(mu)
    omega = sum(mu)/lam_v
    num = 0
    for i in range(len(mu)):
        num = num + np.sqrt(mu[i]/avg_mu)
    return num * h(omega, psi)

def com_lower_bound_1(psi, mu, lam_v):
    omega = sum(mu)/lam_v
    return h(omega, psi)

def com_lower_bound_2(psi, mu, lam_v):
    max_x = max(mu)
    n  = math.floor(sum(mu)/max_x)
    omega = max_x*n/lam_v
    return n*h(omega, psi)

def H_r(mu, r, lam_v, psi):
    # 先排序
    mu = mu.sort(reverse=True)
    omega_T_r = sum(mu[0:r])/lam_v
    return r * h(omega_T_r, psi)

def h(omega, psi):
    if omega > 1:
        return 1/omega + psi/(omega*(omega - 1))
    else:
        return INFINITY

def trans_upper_bound_1(alpha, phi_v, theta_v, beta_d, link_bans, lam):
    data_amount  = (sum(alpha) - theta_v)*beta_d
    ban_amount = sum(link_bans) - phi_v
    return 2*data_amount/ban_amount

def trans_upper_bound_2(alpha, phi_v, theta_v, beta_d, link_bans, lam):
    data_amount = 0
    for i in range(len(alpha)):
        data_amount =  data_amount + max(alpha[i]-lam[i], 0)
    data_amount = data_amount*beta_d
    ban_amount = sum(link_bans) - phi_v
    if ban_amount > 0:
        return 2*data_amount/ban_amount
    else:
        return 0
        print(sum(link_bans))
        raise RuntimeError("ban_amount is %d, should be a positive, phi_v is %d." % (ban_amount, phi_v))

def cal_psi(simga_h, beta_h):
    return (simga_h**2 + beta_h**2)/(2*beta_h**2)

def transform_result(tree):
    """ 将解解析出来 """
    pass

def print_result(x, r):
    print("")


# test functions
def test_hierarchy(tng, max_iteration=40, repeat=5, need_timing=False):
    results = []
    # np.random.seed(seed)
    # np.set_printoptions(formatter={'float':'{:.3f}'.format})

    time_consumption = []

    tree = test_constructor(tng, do_print=False)
    success_count = 0
    start_time = time.time()
    while success_count < repeat:
        tree = test_constructor(tng, do_print=False)
        success = solve_problems(tree, max_repeat=max_iteration, max_success=3 )
        if success == True:
            end_time = time.time()

            time_consumption.append(end_time - start_time)
            start_time = end_time

            results.append(cal_total_time_from_tree(tree))
            success_count =  success_count + 1
    print(results)
    if need_timing == False:
        return results
    else:
        return results, time_consumption

def test_traverse_tree(tree):
    node = traverse_tree(tree, "tng-0-0-1-1")
    print(json.dumps(node,sort_keys=True, indent=4, separators=(',', ': ')))

def test_solver(node):
    solve_problems(node)
    store_tree(tree, 'puppy/results/hierarchy/tng-1-result.json')
    # print(json.dumps(node,sort_keys=True, indent=4, separators=(',', ': ')))
    print(cal_total_time_from_tree(tree))
    print(cal_total_com_time(tree))

def test_gen_group_obj():
    with open('puppy/results/hierarchy/tng-1-result.json', 'r') as f:
        tree = json.load(f)
    node = traverse_tree(tree, "tng-0-0-0-1")
    x = [1.406, 2.359, 0.004, 0.003, 1.996, -0.000, 1.997, -0.000, 2908.063,
       0.530, 6.373, 2296.429]
    obj = gen_obj_for_group(node)
    print("obj(x) is: ",obj(x))

def test_solver_on_dummy():
    np.random.seed()
    with open('puppy/results/hierarchy/tng-1-result.json', 'r') as f:
        tree = json.load(f)
    node = traverse_tree(tree, "tng-0-0")
    # obj = gen_obj_for_nongroup(node)
    cons = gen_cons(node, BETA_D[0])
    children = node['children']
    iteration = 0
    min_fun = 10000000000000
    min_res = None
    
    success = False
    if len(children[0]['children']) > 0: # 如果存在孙子
        print(children[0]['children'])
        obj = gen_obj_for_nongroup(node)
        print("dummy")
    elif children[0]['type'] == TYPE_SERVER:
        obj = gen_obj_for_group(node)
        print('group')
    iteration = 0
    while iteration < 30:
        print("iteration ",iteration)
       
        x0 = gen_feasible_solution(cons, node).x
        # x0 = np.random.rand()
        res = minimize(obj, x0, method='SLSQP', constraints=cons)
        
        success = res.success
        iteration = iteration + 1
        if res.fun < min_fun and res.success == True:
            print(res.fun)
            min_fun = res.fun
            min_res = res
    print("In %s , type is %s:" % (node['name'],node['type']))
    print(min_res)
    x = min_res.x
    print(obj(x))
    tree['solution'] = result_to_dict(res.x, len(children))
    children = node['children']
    r = len(children)
    lam_v = lam_vs(x, r)
    rscs = []
    for v in range(r):
        rscs.append(children[v]['rsc'])
    print("lams:\n",lam_v)
    print("rscs:\n", rscs)
    omega = np.array(rscs)/lam_v
    print("omega:\n",omega)
    k = 0
    psi = cal_psi(SIGMA_H[0], BETA_H[0])
    for child in children:
        print("m*h(omega): ",len(child['children'])*h(omega[k], psi))
        k = k + 1
    omega_0 = sum(rscs)/sum(lam_v)
    print("omega_0:",omega_0)
    print("m*h(omega_0)", len(children)*h(omega_0, psi))
    check_constraints(cons, x)

def test_wolf():
    mu = [8, 4, 1, 5, 2, 2, 2, 4]
    lam = [6.355, 2.000, 0.061, 2.634, 0.793, 0.961, 0.825, 2.371]
    time = 0
    for m in range(len(mu)):
        time = time + f_v(lam_v=lam[m], mu_v=mu[m], sigma_h=SIGMA_H[0], beta_h=BETA_H[0])
    print(time)
    time = 0
    for m in range(len(mu)):
        time = time +f_v2(lam_v=lam[m], mu_v=mu[m], sigma_h=SIGMA_H[0], beta_h=BETA_H[0])
    print(time)


def test_gen_obj(node):
    cons = gen_cons(node, BETA_D[0])
    obj = gen_obj_for_nongroup(node)
    r = len(node['children'])
    x0 = gen_feasible_solution(cons, node, True).x
    print("checking generated feasible solution:")
    check_constraints(cons, x0)
    res = minimize(obj, x0, method='SLSQP', constraints=cons)
    print(res)

def test_gen_feasible(node):
    cons = gen_cons(node, BETA_D[0])
    r = len(node['children'])
    x0 = gen_feasible_solution(cons, node, True).x
    check_constraints(cons, x0)
    result = result_to_dict(x0, r)
    print(json.dumps(result, sort_keys=True, indent=4, separators=(',', ': ')))


def test_gen_cons(node):
    r = len(node['children'])
    x0 = np.zeros(2*r + r**2*2)
    x1 = np.ones(2*r + r**2*2)
    cons = gen_cons(node, BETA_D[0])
    print("If x is zeros")
    check_constraints(cons, x0)
    print("If x is ones")
    check_constraints(cons, x1)

    pass

def test_constructor(tng, do_print=False):
    tree = transform_tng(tng)
    # if do_print:
    #     print(json.dumps(tree,sort_keys=True, indent=4, separators=(',', ': ')))

    tree = reconstruct_tree(tree, 2)
    # if do_print:
    #     print("After reconstruction:")
    #     print(json.dumps(tree,sort_keys=True, indent=4, separators=(',', ': ')))
       
    construct_sub_problems(tree, 'tng-0')
    tree['input_load'] = 0
    tree['output_load'] = 0
    tree['input_ban'] = 0
    tree['output_ban'] = 0
    tree['inlink_ban'] = 0
    tree['outlin_ban'] = 0
    if do_print:
        print("After problem construction:")
        print(json.dumps(tree,sort_keys=True, indent=4, separators=(',', ': ')))
        with open('puppy/results/hierarchy/tng-1.json', 'w') as f:
            json.dump(tree, f, sort_keys=True, indent=4, separators=(',', ': '))
        f.close()


    
    return tree

if __name__=="__main__":
    # 测试用例需要的tng配置是，4个域，每个域4台边缘服务器。
    tng = createATreeGraph()
    tng.print_info()
    np.random.seed()
    np.set_printoptions(formatter={'float':'{:.3f}'.format})

    tree = test_constructor(tng, do_print=False)

    # test_solver_on_dummy()
    # print(f_v(1.844,2, 1, 1))
    # test_gen_group_obj()
    # test_traverse_tree(tree)
    test_solver(tree)
    # test_wolf()
    # tree = load_tree('puppy/results/hierarchy/tng-1-result.json')
    print(cal_total_time_from_tree(tree))
    
    # test_gen_obj(tree)
    # test_gen_feasible(tree)
    # test_gen_cons(tree)