
import numpy as np
from .wolf import *
from ..net_graph import createATreeGraph
from scipy.optimize import minimize



sigma = 1
beta = 1
alpha = (sigma**2 + beta**2)/(2*beta**2)



def f(lam, mu):
    return (1/mu + (sigma**2 + beta**2)*lam / ((2*mu*beta**2)*(mu-lam)))*lam
    
def H(omega):
    return alpha*(2*omega-1)/((omega - 1)**2) + 1
def om(o):
    return 1/o + alpha/(o*(o-1))
    
def obj(x):
    return f(x[0], mu[0]) + f(x[1], mu[1])

def upper1(omega):
    return 2*(1/omega + alpha/(omega*(omega-1)))

def upper2(omega_t):
    if omega_t <= 1:
        return 100
    else:
        return 1/omega_t + alpha/(omega_t*(omega_t-1))

def upper3(mu, task_amount):
    # dummy_o = mu[0]/(mu[0] - min(task_amount,(sum(c) - task_amount)/(1 + np.sqrt(1/11))) )
    dummy_o = mu[0]/min( mu[0] - (sum(c) - task_amount)/(1 + np.sqrt(1/11)), task_amount )
    return om(dummy_o)*(1 + np.sqrt(1/11))

def gen_constraints(mu, task_amount):
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
    return cons

if __name__ == "__main__":
    
    
    c = np.array([11, 1])
    task_amount = 3
    mu = c/beta
    cons = gen_constraints(mu, task_amount)
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