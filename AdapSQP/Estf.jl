## estimate the augmented Lagrange function
function Estf(nlp,nx,sigma,Xi_f,f_k,nabf_k,c_k,G_k,x,lam,nu,mu,CovM,alpha,NewDir)
    # estimate f
    bf_k = f_k+rand(Normal(0,(sigma/Xi_f)^(1/2)))
    bnabf_k = mean(rand(MvNormal(nabf_k,CovM), convert(Int64,floor(Xi_f))),dims = 2)
    bL_mu_k = bf_k+c_k'lam + mu/2*norm(c_k)^2 + nu/2*norm(G_k*(bnabf_k+G_k'lam))^2

    # estimate f_s_k
    x_sk = x+alpha*NewDir[1:nx]
    lam_sk = lam+alpha*NewDir[nx+1:end]
    f_sk, nabf_sk = objgrad(nlp,x_sk)
    c_sk, G_sk = consjac(nlp,x_sk)
    bf_sk = f_sk + rand(Normal(0,(sigma/Xi_f)^(1/2)))
    bnabf_sk = mean(rand(MvNormal(nabf_sk,CovM), convert(Int64,floor(Xi_f))),dims = 2)
    bL_mu_sk = bf_sk+c_sk'*lam_sk + mu/2*norm(c_sk)^2 + nu/2*norm(G_sk*(bnabf_sk+G_sk'*lam_sk))^2
    return bL_mu_k, bL_mu_sk
end
