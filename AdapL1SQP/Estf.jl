## estimate the augmented Lagrange function
function Estf(nlp,nx,sigma,Xi_f,f_k,c_k,x,mu,alpha,NewDir)
    # estimate f
    bf_k = f_k+rand(Normal(0,(sigma/Xi_f)^(1/2)))
    bL_mu_k = bf_k + mu*norm(c_k,1)
    # estimate f_s_k
    x_sk = x+alpha*NewDir[1:nx]
    f_sk, nabf_sk = objgrad(nlp,x_sk)
    c_sk, G_sk = consjac(nlp,x_sk)
    bf_sk = f_sk + rand(Normal(0,(sigma/Xi_f)^(1/2)))
    bL_mu_sk = bf_sk + mu*norm(c_sk,1)
    return bL_mu_k, bL_mu_sk
end
