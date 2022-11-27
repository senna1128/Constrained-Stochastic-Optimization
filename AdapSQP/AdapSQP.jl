include("EstLandM.jl")
include("Estf.jl")
## Implement NonAdaptive SQP for a single problem
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

function AdapSQP(nlp,sigma,Max_Iter,EPS_Step,EPS_Res,nu,mu,epsilon,beta,rho,alpha_max,kap_grad,kap_f,p_grad,p_f,C_grad)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon

    # Initialize
    eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
    # evaluate objective, gradient, Hessian
    f_k, nabf_k = objgrad(nlp, X[end])
    nab2f_k = hess(nlp, X[end])
    # evaluate constraint and Jacobian
    c_k, G_k = consjac(nlp, X[end])
    # Lagrangian gradient and Hessian
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    # KKT residual
    KKT = [norm([nab_xL_k; c_k])]
    # Covariance matrix
    CovM = sigma*(Diagonal(ones(nx))+ones(nx,nx))
    # Indicator of convergence and singularity
    IdCon, IdSing = 1, 0
    # Other initial parameters
    Xi, Xi_f, alpha_k, bnabL_mu_k, Quant = 1, 1, alpha_max, zeros(nx+nlam), 0
    Count_G,Count_F,Alpha,nab_x2c_k = [],[],[],Array{Array}(undef, nlam)

    # Time
    Time = time()
    while eps>EPS_Step && KKT[end]>EPS_Res && k<Max_Iter
        ## Obtain est of bnab_xL_k and bM_k
        for i = 1:nlam
            nab_x2c_k[i] = hess(nlp,X[end],obj_weight=1.0,1:nlam.==i)-nab2f_k
        end
        bnab_xL_k, bM_k, Xi = EstLandM(nx,nlam,Xi,G_k,c_k,nab_xL_k,nab_x2L_k,nab_x2c_k,sigma,CovM,nu,alpha_k,kap_grad,p_grad,C_grad,rho)
        push!(Count_G, Xi)

        ## Compute Newton Direction
        try
            FullH = hcat(vcat(Diagonal(ones(nx)),G_k), vcat(G_k',zeros(nlam,nlam)))
            FullG = vcat(bnab_xL_k,c_k)
            NewDir = lu(FullH)\-FullG
            NewDir[nx+1:end] = lu(G_k*G_k')\-(G_k*bnab_xL_k + bM_k'NewDir[1:nx])
        catch
            IdSing = 1
        end
        if IdSing == 1
            return [],[],[],[],[],[],0,IdCon,IdSing
        else
            # update mu
            while true
                bnabL_mu_k = [bnab_xL_k+nu*bM_k*G_k*bnab_xL_k+mu*G_k'c_k; c_k+nu*G_k*G_k'*G_k*bnab_xL_k]
                Quant = (bnabL_mu_k'NewDir)[1]
                if mu > 1e8
                    break
                elseif Quant > -min(nu,1)/2*norm([NewDir[1:nx];G_k*bnab_xL_k])^2 || norm(c_k) > norm(bnabL_mu_k)
                    mu *= rho
                else
                    break
                end
            end

            # Estimate function value
            Quant_com = min((kap_f*alpha_k^2*Quant)^2, epsilon^2, 1)
            Xi_f = min(C_grad*log(8*nx/p_f)/Quant_com, 1e5)
            push!(Count_F, Xi_f)
            bL_mu_k, bL_mu_sk = Estf(nlp,nx,sigma,Xi_f,f_k,nabf_k,c_k,G_k,X[end],Lam[end],nu,mu,CovM,alpha_k,NewDir)

            if bL_mu_sk <= bL_mu_k + alpha_k*beta*Quant
                push!(X, X[end]+alpha_k*NewDir[1:nx])
                push!(Lam, Lam[end]+ alpha_k*NewDir[nx+1:end])
                push!(Alpha,alpha_k)
                eps, k = norm(alpha_k*NewDir), k+1
                f_k, nabf_k = objgrad(nlp, X[end])
                nab2f_k = hess(nlp, X[end])
                c_k, G_k = consjac(nlp, X[end])
                nab_xL_k = nabf_k + G_k'Lam[end]
                nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
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
