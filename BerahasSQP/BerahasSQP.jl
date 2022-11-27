include("Proj.jl")
## Implement NonAdaptive SQP for a single problem
# Input
### nlp: problem
### Step: Step size
### varsigma: standard deviation of approximation
### Max_Iter: maximum iteration
### EPS: minimum of difference
### tau, epsilon, sigma, xi, theta
### IdConst: indication of constant stepsize
#    IdConst == 1: constant step size
#    IdConst == 0: decay step size
# Output
### X: X iteration sequence
### Lam: Lam iteration sequence
### KKT: KKT residual iteration sequence
### Time: consuming time
### IdCon: indicator of whether convergence
### IdSing: indicator of singular

function BerahasSQP(nlp,Step,varsigma,Max_Iter,EPS_Step,EPS_Res,tau,epsilon,sigma,xi,theta,IdConst)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon

    # Initialize
    eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
    c_k, G_k = consjac(nlp, X[end])
    nabf_k = grad(nlp, X[end])
    KKT = [norm([nabf_k + G_k'Lam[end]; c_k])]
    CovM = varsigma*(Diagonal(ones(nx)) + ones(nx, nx))
    L_k = norm(hess(nlp, X[end]))+1
    Gamma_k = 1e5 * norm(jac(nlp,X[end]+1e-5*ones(nx)) - jac(nlp, X[end]))
    IdCon, IdSing = 1, 0

    # Time
    Time = time()
    while eps>EPS_Step && KKT[end]>EPS_Res && k<Max_Iter
        ## Obtain est of gradient of Lagrange
        bnabf_k = rand(MvNormal(nabf_k, CovM))
        ## Compute Newton Direction
        try
            FullH = hcat(vcat(Diagonal(ones(nx)), G_k), vcat(G_k', zeros(nlam, nlam)))
            FullG = vcat(bnabf_k, c_k)
            NewDir = lu(FullH)\-FullG
        catch
            IdSing = 1
        end
        if IdSing == 1
            return [], [], [], 0, IdCon, IdSing
        elseif norm(NewDir[1:nx]) < 1e-8
            k += 1
            continue
        else
            d2_k = norm(NewDir[1:nx])^2
            IntQuan_1 = bnabf_k'NewDir[1:nx] + d2_k
            # compute tau_trial
            if IntQuan_1 > 0
                tau_trial = (1-sigma)*norm(c_k, 1)/IntQuan_1
                if tau > tau_trial
                    tau = (1-epsilon)*tau_trial
                end
            end
            # compute xi_trial
            IntQuan_2 = -tau*(bnabf_k'NewDir[1:nx] + 0.5*d2_k) + norm(c_k,1)
            xi_trial = IntQuan_2/(tau * d2_k)
            # compute xi_k
            if xi > xi_trial
                xi = (1-epsilon)*xi_trial
            end
            # set boundary of stepsize
            beta_k = IdConst*Step+(1-IdConst)/(k+1)^Step
            Est_L_k = tau*L_k+Gamma_k
            hat_alpha_k_init = beta_k*IntQuan_2/(Est_L_k*d2_k)
            tilde_alpha_k_init = hat_alpha_k_init - 4*norm(c_k,1)/(Est_L_k*d2_k)
            LB = beta_k*xi*tau/Est_L_k
            RB = LB + theta*beta_k^2
            hat_alpha_k = Proj(hat_alpha_k_init, LB, RB)
            tilde_alpha_k = Proj(tilde_alpha_k_init, LB, RB)
            if hat_alpha_k < 1
                alpha_k = hat_alpha_k
            elseif tilde_alpha_k <= 1 && 1 <= hat_alpha_k
                alpha_k = 1
            else
                alpha_k = tilde_alpha_k
            end
            push!(X, X[end]+alpha_k*NewDir[1:nx])
            push!(Lam, NewDir[nx+1:end])
            eps, k = norm(alpha_k*NewDir), k+1
            c_k, G_k = consjac(nlp, X[end])
            nabf_k= grad(nlp, X[end])
            push!(KKT, norm([nabf_k + G_k'Lam[end]; c_k]))
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [], [], [], Time, 0, 0
    else
        return X, Lam, KKT, Time, 1, 0
    end

end
