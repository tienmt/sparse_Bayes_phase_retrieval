#source('/Users/thetm/Dropbox/Apps/Overleaf/sparse phase retrival/Rcodes/algorithms.R')
library(Rcpp)
Rcpp::sourceCpp('C:/Users/thmai/Documents//test.cpp')
# sigma: measurement noise
m = 500   # m: number of observations
p = 100    # p: dimension of the signal x^*
k = 10    # k: sparsity level
Iters = 30000
burnin = 1000
my_sigma = 1
their_iter = 5000
my_lambda = 1
lmc = mala = mirrordecent = accept = c()
for (ss in 1:100) {
  
  A = matrix(rnorm(m*p), nrow = m, ncol = p); tA = t(A)
  beta0 = rnorm( p, sd = .0)
  ind = 1:k
  beta0[ind] = rnorm( k, sd = 1)
  beta0 = beta0 / sqrt(sum(beta0^2))
  y = as.vector((A%*%beta0)^2) + rnorm(m, sd = my_sigma)
  
  ini1 = sort(t(A^2) %*% y, index.return = TRUE)$ix[p]
  res1 = hwf( A, y, ini = ini1, step = 0.1, beta = 1e-14, 
              iteration = their_iter, iterates = T, x_star = 0)
  
  res2 = mirror_eg( A, y, ini = ini1, step = 0.1, beta = 1e-14, 
                    iteration = their_iter, iterates = TRUE, x_star = 0)
  ### MALA
  tau = .1  # in the prior
  b_mala = matrix( 0 ,nrow = p); a = 0  ; 
  M = res1[their_iter,]; M[M<0.001] <- 0
  h = 1/(p)^3.8 
  for(s in 1:Iters){
    Xm = eigenMapMatMult( A, M )  
    my_grad = 4* eigenMapMatMult (tA , (Xm^2 -y ) *Xm ) * my_lambda
    tam = M - h*my_grad - h*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h)*rnorm(p)
    Xtam = eigenMapMatMult( A, tam )  
    my_grad_tam = 4* eigenMapMatMult (tA, ((Xtam^2 - y) * Xtam) ) * my_lambda
    pro.tam = - sum(  my_lambda * (y - Xtam^2)^2 ) - sum(2*log(tau^2 + tam^2))
    pro.M = - sum( my_lambda * (y - Xm^2 )^2 ) - sum(2*log(tau^2 + M^2))
    tran.m = -sum((M - tam + h*my_grad_tam + h*sum(2*log(tau^2 + tam^2)) )^2)/(4*h)
    tran.tam = -sum((tam - M + h*my_grad + h*sum(2*log(tau^2 + M^2)) )^2)/(4*h)
    pro.trans = pro.tam + tran.m - pro.M - tran.tam
    if(log(runif(1)) <= pro.trans){
      M = tam;   a = a+1  } 
    if (s>burnin)b_mala = b_mala + M/(Iters-burnin)
  } ; 
  print( accept[ss] <- a/Iters)
  
  ### LMC
  b_lmc = matrix( 0 ,nrow = p); 
  h_lmc =  1/(p)^4
  M = res1[their_iter,] ;  M[M<0.001] <- 0
  for(s in 1:Iters){
    Xm = eigenMapMatMult( A, M ) 
    my_grad = my_lambda *eigenMapMatMult (tA , (Xm^2 -y ) *Xm )
    M = M - h_lmc*my_grad - h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
    if (s>burnin)b_lmc = b_lmc + M/(Iters-burnin)}
  print(ss)
  mala[ss] = mean((b_mala -beta0)^2 )
  lmc[ss] = mean((b_lmc -beta0)^2 )
  mirrordecent[ss] = mean((beta0 -res2[their_iter,])^2) 
}
#rm(A,res1,res2)
#setwd("/Users/thetm/Dropbox/Apps/Overleaf/sparse phase retrival/Rcodes/outsimu/")
#save.image(file = 'm500p100k10_lambda_1_2.rda')
