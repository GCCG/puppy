setwd("C:/Users/DELL/OneDrive/ÎÄµµ/WorkSpace/S_Code/puppy/results")

library(Rdonlp2)

A <- as.matrix(read.txt("A.txt", header = FALSE)) 
lb <- as.matrix(read.txt("lb.txt", header = FALSE))
ub <- as.matrix(read.txt("ub.txt", header = FALSE))
x0 <- as.matrix(read.txt("x0.txt", header = FALSE))
pal <- as.matrix(read.txt("pa_l.txt", header = FALSE))
pau <- as.matrix(read.txt("pa_u.txt", header = FALSE))
mu <- as.matrix(read.txt("mu.txt", header = FALSE))
omega <- read.txt("omega.txt", header = FALSE))
para <- read.txt("para.txt", header = FALSE))
c <- read.txt("c.txt", header = FALSE))

server_num <- para

fn = function(x){
	for (j in (
	
	

