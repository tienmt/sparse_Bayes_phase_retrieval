library(tictoc)
library(dslabs)
mnist <- read_mnist()
#
#  one need to download the code from:
# Wu, F. and Rebeschini, P. (2023). Nearly minimax-optimal rates for noisy sparse phase retrieval via
# early-stopped mirror descent. Information and Inference: A Journal of the IMA, 12(2):633–713
#
#source('/Users/thetm/Dropbox/Apps/Overleaf/sparse phase retrival/Rcodes/algorithms.R')
Iters = 30000
burnin = 1000
Rcpp::sourceCpp('/Users/thetm/Dropbox/Apps/Overleaf/sparse phase retrival/Rcodes/test.cpp')
# sigma: measurement noise
m = 4000   # m: number of observations

i <- 36 # i = 5 for digit 4 -c(1:6,23:28), 26:5 ;
# 36 = 2 -c(1:4,25:28), 25:4 ; 
beta_true <-mnist$test$images[i,]
beta_true = beta_true / sqrt(sum(beta_true^2))
beta0 <- matrix(beta_true, nrow=28)[ -c(1:4,25:28), 25:4   ]
par(mfrow=c(1,4),mar=c(1,1,1,1))
image(1:nrow(beta0), 1:ncol(beta0), beta0,  main='true image',
      col = gray(seq(0, 1, 0.05)), xlab = "", ylab="",xaxt='n',yaxt='n')
p = prod(dim(beta0)) 
A = matrix(rnorm(m*p), nrow = m, ncol = p); tA = t(A)
my_sigma = 1
y = as.vector((A%*%c(beta0))^2) + rnorm(m, sd = my_sigma)

their_iter = 5000
ini1 = sort(t(A^2) %*% y, index.return = TRUE)$ix[p]
res1 = hwf( A, y, ini = ini1, step = 0.1, beta = 1e-6, 
            iteration = their_iter, iterates = T, x_star = 0)
image(1:nrow(beta0), 1:ncol(beta0), matrix(res1[their_iter,] , nrow=nrow(beta0) ), main='HWF',xaxt='n',yaxt='n' , col = gray(seq(0, 1, 0.05)), xlab = "", ylab="")
res2 = mirror_eg( A, y, ini = ini1, step = 0.1, beta = 1e-2, 
                  iteration = 5000, iterates = TRUE, x_star = 0)
image(1:nrow(beta0), 1:ncol(beta0), matrix(res2[their_iter ,], nrow=nrow(beta0) ), main='MD',xaxt='n',yaxt='n' , col = gray(seq(0, 1, 0.05)), xlab = "", ylab="")


### MALA
tau = .1  # in the prior
b_mala = matrix( 0 ,nrow = p); a = 0  ; 
M = res1[their_iter,]; M[M<0.001] <- 0
h = 1/(p)^3.4 
tic()
for(s in 1:Iters){
  Xm = eigenMapMatMult( A, M ) 
  my_grad = 4* eigenMapMatMult (tA , ((Xm^2 -y ) *Xm) )
  tam = M - h*my_grad - h*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h)*rnorm(p)
  
  Xtam = eigenMapMatMult( A, tam ) ;   
  my_grad_tam = 4* eigenMapMatMult (tA, ((Xtam^2 - y) * Xtam) )
  pro.tam = - sum( (y - Xtam^2)^2 ) - sum(2*log(tau^2 + tam^2))
  pro.M = - sum( (y - Xm^2 )^2 ) - sum(2*log(tau^2 + M^2))
  
  tran.m = -sum((M - tam + h*my_grad_tam + h*sum(2*log(tau^2 + tam^2)) )^2)/(4*h)
  tran.tam = -sum((tam - M + h*my_grad + h*sum(2*log(tau^2 + M^2)) )^2)/(4*h)
  pro.trans = pro.tam + tran.m - pro.M - tran.tam
  if(log(runif(1)) <= pro.trans){
    M = tam;   a = a+1  } ;  if (s%%5000==0){print(s) }
  if (s>burnin)b_mala = b_mala + M/(Iters-burnin)
} ; a/Iters
toc()
image(1:nrow(beta0), 1:ncol(beta0), matrix(b_mala, nrow=nrow(beta0) ), main='MALA',xaxt='n',yaxt='n' , col = gray(seq(0, 1, 0.05)), xlab = "", ylab="")

### LMC
b_lmc = matrix( 0 ,nrow = p); 
h_lmc =  1/(p)^4
M = res1[their_iter,] ;  M[M<0.001] <- 0
tic()
for(s in 1:Iters){
  Xm = eigenMapMatMult( A, M ); my_grad =  eigenMapMatMult( tA , (Xm^2 -y ) *Xm )
  M = M - h_lmc*my_grad - h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
  if (s>burnin)b_lmc = b_lmc + M/(Iters-burnin)}
toc()
image(1:nrow(beta0), 1:ncol(beta0), matrix(b_lmc, nrow=nrow(beta0) ), main='LMC',xaxt='n',yaxt='n' , col = gray(seq(0, 1, 0.05)), xlab = "", ylab="")

c( mean((b_mala -c(beta0))^2 ), mean((b_lmc -c(beta0))^2 ), mean((c(beta0) -res1[their_iter,])^2) , mean((c(beta0) -res2[their_iter,])^2) )
c( mean( abs(b_mala -c(beta0)) ),mean( abs(b_lmc -c(beta0)) ),mean( abs(res1[their_iter,] -c(beta0)) ), mean( abs(res2[their_iter,] -c(beta0)) ))
a/Iters

