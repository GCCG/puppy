library(Rdonlp2)
p = c(10,10)    #迭代初始值
par.l = c(-100,-100); par.u = c(100,100) # 目标值域
# 目标
fn = function(x){
  x[1]^2*sin(x[2])+x[2]^2*cos(x[1])
}
# 行构成线性约束
A = matrix(c(1,1,3,-1),2,byrow=TRUE)
lin.l = c(2,1); lin.u = c(+Inf,3)  
# 构成非线性约束 
nlcon1 = function(x){
  x[1]*x[2]
} 
nlcon2 = function(x){
  sin(x[1])*cos(x[2])
}
nlin.l = c(2,-Inf)
nlin.u = c(2,0.6)  
# 求解
ret = donlp2(p, fn, par.u=par.u, par.l=par.l,A, lin.l=lin.l, lin.u=lin.u, nlin = list(nlcon1,nlcon2), nlin.u = nlin.u, nlin.l = nlin.l)
