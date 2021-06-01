# from cvxpy import *
# from sympy import *
import numpy as np
from .wolf import *
from ..net_graph import createATreeGraph
from scipy.optimize import minimize


# func_list = []
# h = 4
# for i in range(5):
#     y = lambda x, i=i: x + i + h
#     func_list.append(y)
# for e in func_list:
#     print(e)
#     print(eval(e))
#     print(e(3))


# coding=utf-8


# A = np.zeros((6,3))
# A[0,0] = 1
# A[1,0] = -1
# A[2,1] = 1
# A[3,1] = -1
# A[4,2] = 1
# A[5,2] = -1
# # demo 2
# #计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
# def fun(args):
#     a,b,c,d=args
#     # v=lambda x: (a+x[0])/(b+x[1]) -c*x[0]+d*x[2]
#     def v(x):
#         y = 0
#         for i in range(3):
#             y = y + x[0]
#         return (a+x[0])/(b+x[1]) -c*x[0]+d*x[2] + y
#     return v
# def con(args):
#     # 约束条件 分为eq 和ineq
#     #eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0  
#     x1min, x1max, x2min, x2max,x3min,x3max = args
#     bia = [x1min, x1max, x2min, x2max,x3min,x3max]
#     # cons = [{'type': 'ineq', 'fun': lambda x: x[0] - x1min},\
#     #           {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},\
#     #          {'type': 'ineq', 'fun': lambda x: x[1] - x2min},\
#     #             {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},\
#     #         {'type': 'ineq', 'fun': lambda x: x[2] - x3min},\
#     #          {'type': 'ineq', 'fun': lambda x: -x[2] + x3max}]
#     def s(x):
#         return -x[2] + x3max 
#     cons = [{'type': 'ineq', 'fun': lambda x: np.dot(A[0],x) - bia[0]},\
#               {'type': 'ineq', 'fun': lambda x: np.dot(A[1],x) + bia[1]},\
#              {'type': 'ineq', 'fun': lambda x: np.dot(A[2],x) - bia[2]},\
#                 {'type': 'ineq', 'fun': lambda x: np.dot(A[3],x) + bia[3]},\
#             {'type': 'ineq', 'fun': lambda x: np.dot(A[4],x) - bia[4]},\
#              {'type': 'ineq', 'fun': lambda x: np.dot(A[5],x) + bia[5]}]
#     return cons

def tet_partial_iterative_funcion():
    pass

def test_random():
    def obj(x):
        index = np.random.randint(0, len(x))
        return x[index]
    x0 = np.random.rand(10)

    res = res = minimize(obj, x0, method='SLSQP',constraints=[])
    print(res)
 
if __name__ == "__main__":
    # #定义常量值
    # args = (2,1,3,4)  #a,b,c,d
    # #设置参数范围/约束条件
    # args1 = (0.1,0.9,0.1, 0.9,0.1,0.9)  #x1min, x1max, x2min, x2max
    # cons = con(args1)
    # #设置初始猜测值  
    # x0 = np.asarray((0.5,0.5,0.5))
    """ np.set_printoptions(formatter={'float':'{:.3f}'.format})
    np.random.seed(9)

    ng = createATreeGraph()
    args = gen_args(ng)
    ng.print_info()

    # test_random()
    # cons = []
    # cons = constraints_for_links(ng)
    # cons = constraints_for_offloading(ng) + cons
    # cons = constraints_for_path_ban(ng) + cons
    # cons = constraints_for_receiver(ng) + cons
    # cons = constraints_for_sender(ng) + cons

    cons = gen_constraints(ng)
    # cons =  gen_intact_constraints(ng)
    s_len = len(ng.getServerList())
    # x0 = np.ones(2*s_len**2)*0.5

    resutl_for_slsqp = []
    def callback(xk):
        resutl_for_slsqp.append(xk)

    # def callbackK(xk):
    #     print(xk)
    iteration = 20
    success = 0
    for i in range(iteration):
        x0 = gen_feasible_solution_exterior(cons, args, cons_mode=0).x
        # x0 = np.random.rand(2*s_len**2)
        res = minimize(gen_objective(args), x0, method='SLSQP',constraints=cons)
        print("Result: %s,message: %s" %(res.success, res.message))
        if res.success == True:
            success = success + 1
            print(res)
            check_constraints(cons, res.x)
            # A, B = reshape_x(res.x, s_len)
            # check_constraints(cons, res.x)
            # print("offloading decisions:")
            # print(A)
            # print("bandwidth:")
            # print(B)
            print_result(res.x, s_len)
    print("Iteration %d times, %d succeed, ratio %F" %(iteration, success, success/iteration)) """
    
    # check_constraints(cons, res.x)
    # A, B = reshape_x(res.x, s_len)
    # print("offloading decisions:")
    # print(A)
    # print("bandwidth:")
    # print(B)

    c = np.array([11, 1])
    sigma = 1
    beta = 1
    alpha = (sigma**2 + beta**2)/(2*beta**2)
    task_amount = 10

    mu = c/beta
    def f(lam, mu):
        return (1/mu + (sigma**2 + beta**2)*lam / ((2*mu*beta**2)*(mu-lam)))*lam
    
    def H(omega):
        return alpha*(2*omega-1)/((omega - 1)**2) + 1
    def om(o):
        return 1/o + alpha/(o*(o-1))
    
    def obj(x):
        return f(x[0], mu[0]) + f(x[1], mu[1])

    def h(omega):
        return 1/omega + alpha/(omega*(omega-1))
    def upper1(omega):
        return 2*(1/omega + alpha/(omega*(omega-1)))
    def upper2(omega_t):
        if omega_t <= 1:
            return 100
        else:
            return 1/omega_t + alpha/(omega_t*(omega_t-1))
    def upper3(mu, task_amount):
        dummy_o = mu[0]/(mu[0] - (sum(c) - task_amount)/(1 + np.sqrt(1/11)))
        return om(dummy_o)*(1+ np.sqrt(1/11))
    # def upper3(omega, mu):
    #     avg = np.average(mu)
    #     para = np.sum(np.abs(mu-avg))/avg
    #     print("para:",para)
    #     return para*(1/omega + alpha/(omega*(omega-1)))
    
    # def upper4(c, task_amount, mu):
    #     min_c = min(c)
    #     unit = task_amount/(np.sqrt(c[0]/min_c)+np.sqrt(c[1]/min_c))
    #     print("unit:",unit)
    #     print("task_amount:",unit * np.sqrt(c[0]/min_c)+unit * np.sqrt(c[1]/min_c))
    #     omega_1 = mu[0]/(unit * np.sqrt(c[0]/min_c))
    #     omega_2 = mu[1]/(unit * np.sqrt(c[1]/min_c))
    #     print("omega_1, omega_2:", omega_1,omega_2)
    #     return h(omega_1) + h(omega_2)
    
    cons = []
    con1 = {}
    con1['fun'] = lambda x: x[0] + x[1] - task_amount
    con1['type'] = 'eq'
    con1['info'] = ''
    con2 = {}
    con2['fun'] = lambda x: x[0]
    con2['type'] = 'ineq'
    con2['info'] = ''
    con3 = {}
    con3['fun'] = lambda x: x[1]
    con3['type'] = 'ineq'
    con3['info'] = ''
    con4 = {}
    con4['fun'] = lambda x: mu[0] - x[0] - 0.0001
    con4['type'] = 'ineq'
    con4['info'] = ''
    con5 = {}
    con5['fun'] = lambda x: mu[1] - x[1] - 0.0001
    con5['type'] = 'ineq'
    con5['info'] = ''
    cons.append(con1)
    cons.append(con2)
    cons.append(con3)
    cons.append(con4)
    cons.append(con5)

    res = minimize(obj, [3,3], method='SLSQP', constraints=cons)
    print(res)
    omega = mu/res.x
    omega_0 = sum(c)/task_amount
    print("omega:\n",omega)
    print("omega_0:",omega_0)
    print("(omega_1-1)/(omega_0-1)",(omega[0]-1)/(omega_0-1))
    print("(omega_2-1)/(omega_0-1)",(omega[1]-1)/(omega_0-1))
    print("(omega_1-1)/(omega_2-1)",(omega[0]-1)/(omega[1]-1))
    print("x1/x2:",c[0]/c[1])
    print("H(o1)/H(o2):",H(omega[0])/H(omega[1]))
    print("f(x1):\n",f(res.x[0],mu[0]))
    print("f(x2):\n",f(res.x[1],mu[1]))
    print("f(x1)/f(x2)",f(res.x[0],mu[0])/f(res.x[1],mu[1]))
    print("om(o1)+om(o2)",om(omega[0])+om(omega[1]))
    print(om(omega[0]), om(omega[1]))
    print("upperbound1:",upper1(np.sum(mu)/task_amount))
    
    print("upperbound2:",upper2(np.max(mu)/task_amount))
    # print("obj/upper1",res.fun/upper1(np.sum(mu)/task_amount))
    # print("om(omega[0])/upper1:",om(omega[0])/om(np.sum(mu)/task_amount))
    # print("om(omega[1])/upper1:",om(omega[1])/om(np.sum(mu)/task_amount))
    print("upper3:",upper3(mu=mu, task_amount=task_amount))
    
    x= res.x
    check_constraints(cons, x)
    print("f0/x0",f(x[0],mu[0])/x[0])
    print("f1/x1",f(x[1],mu[1])/x[1])
    print("f0",f(x[0],mu[0]))
    print("f1",f(x[1],mu[1]))
    print("fake:",om((12+6)/(task_amount+6*11/12)))
    






# def grad(f,x):
#     g = []
#     for e in x:
#         g.append(diff(f,e))
#     return g

# x = []
# for i in range(3):
#     for j in range(3):
#         x.append(symbols("x"+str(i)+str(j)))
# print("grad:",grad(np.dot(x,x), x))

# x = []
# x.append(symbols('x'))
# x.append(symbols("y"))
# x.append(symbols("z"))
# def grad(f,x):
#         return [diff(f, x[0]),diff(f,x[1]),diff(f,x[2])]
# def func(x):
#     return np.sum([x[0]**2,x[1]**2,x[2]**2])
# f = x[0]**2 + x[1]**2 +x[2]**2
# print(grad(func(x), x))


# def func(x):
#     # x[1]**2 + x[0]**2
#     # y = x[0]-x[1] + z
#     # return square(x[0] - x[1])
#     return sum(square(x))

# x = Variable((2,1))
# z = x.T.__mul__(x)
# print("x.T@x's shape:", z.shape)
# constraints = [x[0] + x[1] == 1, x[0] - x[1] >=1]
# obj = Minimize(func(x))
# prob = Problem(obj, constraints)
# prob.solve()
# print("status:",prob.status)
# print("optimal value:", prob.value)
# print("optimal var:", x[0].value, x[1].value)