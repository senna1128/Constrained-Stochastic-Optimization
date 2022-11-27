include("Estgradf.jl")
include("Estf.jl")
## Implement Adaptive SQP for a single problem
# Input
### nlp: problem
### Step: Step size
### varsigma: standard deviation of approximation
### Max_Iter: maximum iteration
### EPS: minimum of difference
### nu,mu,epsilon,beta,rho,alpha_max,
### kap_grad,kap_f,p_grad,p_f,C_grad
# Output
### X: X iteration sequence
### Lam: Lam iteration sequence
### KKT: KKT residual iteration sequence
### Time: consuming time
### IdCon: indicator of whether convergence
### IdSing: indicator of singular

function AdapL1SQP(nlp,sigma,Max_Iter,EPS_Step,EPS_Res,mu,epsilon,beta,rho,alpha_max,kap_grad,kap_f,p_grad,p_f,C_grad)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon

    # Initialize
    eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
    # evaluate objective, gradient, Hessian
    f_k, nabf_k = objgrad(nlp, X[end])
    # evaluate constraint and Jacobian
    c_k, G_k = consjac(nlp, X[end])
    # Lagrangian gradient
    nab_xL_k = nabf_k + G_k'Lam[end]
    # KKT residual
    KKT = [norm([nab_xL_k; c_k])]
    # Covariance matrix
    CovM = sigma*(Diagonal(ones(nx))+ones(nx,nx))
    # Indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # Other initial parameters
    Xi, Xi_f, alpha_k, Quant = 1, 1, alpha_max, 0
    Count_G, Count_F, Alpha = [], [], []

    # Time
    Time = time()
    while eps>EPS_Step && KKT[end]>EPS_Res && k<Max_Iter
        ## Obtain est of bnabf_k and bnab_xL_k
        bnabf_k, bnab_xL_k, Xi = Estgradf(nx,Xi,nabf_k,c_k,nab_xL_k,CovM,alpha_k,kap_grad,p_grad,C_grad,rho)
        push!(Count_G, Xi)

        ## Compute Newton Direction
        try
            FullH = hcat(vcat(Diagonal(ones(nx)),G_k), vcat(G_k',zeros(nlam,nlam)))
            FullG = vcat(bnab_xL_k,c_k)
            NewDir = lu(FullH)\-FullG
        catch
            IdSing = 1
        end
        if IdSing == 1
            return [],[],[],[],[],[],0,IdCon,IdSing
        else
            # update mu
            mu = max((bnabf_k'NewDir[1:nx])[1]/(rho-1)/norm(c_k,1), mu)
            if isnan(mu)
                break
            end

            # Estimate function value
            Quant = (bnabf_k'NewDir[1:nx])[1] - mu*norm(c_k,1)
            Quant_com = min((kap_f*alpha_k^2*Quant)^2, epsilon^2, 1)
            Xi_f = min(C_grad*log(8*nx/p_f)/Quant_com, 1e5)
            if isnan(Xi_f)
                return [],[],[],[],[],[],0,0,IdSing
            end
            push!(Count_F, Xi_f)
            bL_mu_k, bL_mu_sk = Estf(nlp,nx,sigma,Xi_f,f_k,c_k,X[end],mu,alpha_k,NewDir)

            if bL_mu_sk <= bL_mu_k + alpha_k*beta*Quant
                push!(X, X[end]+alpha_k*NewDir[1:nx])
                push!(Lam, Lam[end]+ alpha_k*NewDir[nx+1:end])
                push!(Alpha,alpha_k)
                eps, k = norm(alpha_k*NewDir), k+1
                f_k, nabf_k = objgrad(nlp, X[end])
                c_k, G_k = consjac(nlp, X[end])
                nab_xL_k = nabf_k + G_k'Lam[end]
                push!(KKT, norm([nab_xL_k; c_k]))

                if -alpha_k*beta*Quant >= epsilon
                    epsilon *= rho
                    alpha_k = min(alpha_max, rho*alpha_k)
                else
                    epsilon /= rho
                    alpha_k = min(alpha_max, rho*alpha_k)
                end
            else
                k += 1
                alpha_k/= rho
                epsilon /= rho
            end
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [],[],[],[],[],[],Time,0,0
    else
        return X,Lam,KKT,Count_G,Count_F,Alpha,Time,1,0
    end
end
