# from cvxpy import *
# from sympy import *
import numpy as np
from .gen_problem import *
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
 
if __name__ == "__main__":
    # #定义常量值
    # args = (2,1,3,4)  #a,b,c,d
    # #设置参数范围/约束条件
    # args1 = (0.1,0.9,0.1, 0.9,0.1,0.9)  #x1min, x1max, x2min, x2max
    # cons = con(args1)
    # #设置初始猜测值  
    # x0 = np.asarray((0.5,0.5,0.5))
    np.set_printoptions(formatter={'float':'{:.3f}'.format})
    np.random.seed(9)

    ng = createATreeGraph()
    args = gen_args(ng)


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
            # check_constraints(cons, res.x)
            A, B = reshape_x(res.x, s_len)
            check_constraints(cons, res.x)
            print("offloading decisions:")
            print(A)
            print("bandwidth:")
            print(B)
    print("Iteration %d times, %d succeed, ratio %F" %(iteration, success, success/iteration))
    
    # check_constraints(cons, res.x)
    # A, B = reshape_x(res.x, s_len)
    # print("offloading decisions:")
    # print(A)
    # print("bandwidth:")
    # print(B)




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