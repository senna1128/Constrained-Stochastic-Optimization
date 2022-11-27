## Implement NonAdaptive SQP for a single problem
# Input
### nlp: problem
### Step: Step size
### sigma: standard deviation of approximation
### Max_Iter: maximum iteration
### EPS: minimum of difference
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

function NonAdapSQP(nlp, Step, sigma, Max_Iter, EPS_Step, EPS_Res, IdConst)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon

    # Initialize
    eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
    c_k, G_k = consjac(nlp, X[end])
    nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    KKT = [norm([nab_xL_k; c_k])]
    CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
    IdCon, IdSing = 1, 0

    # Time
    Time = time()
    while eps>EPS_Step && KKT[end]>EPS_Res && k<Max_Iter
        ## Obtain est of gradient of Lagrange
        bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
        new_bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
        ## Obtain est of Hessian of Lagrange
        Delta = rand(Normal(0,sigma^(1/2)), nx, nx)
        bnab_x2L_k = Hermitian(nab_x2L_k, :L) + (Delta + Delta')/2
        ## Compute M_k and T_k
        bT_k = zeros(nx, nlam)
        for i = 1:nlam
            bT_k[:, i] = Hermitian(hess(nlp,X[end],obj_weight=1.0,1:nlam.==i)-nab2f_k,:L) * new_bnab_xL_k
        end
        bM_k = bnab_x2L_k*G_k' + bT_k
        ## Compute Newton Direction
        try
            FullH = hcat(vcat(Diagonal(ones(nx)), G_k), vcat(G_k', zeros(nlam, nlam)))
            FullG = vcat(bnab_xL_k, c_k)
            NewDir = lu(FullH)\-FullG
            NewDir[nx+1:end] = lu(G_k*G_k')\-(G_k*bnab_xL_k + bM_k'NewDir[1:nx])
        catch
            IdSing = 1
        end
        if IdSing == 1
            return [], [], [], 0, IdCon, IdSing
        else
            Stepsize = IdConst*Step+(1-IdConst)/(k+1)^Step
            push!(X, X[end] + Stepsize*NewDir[1:nx])
            push!(Lam, Lam[end] + Stepsize*NewDir[nx+1:end])
            eps, k = norm(Stepsize * NewDir), k+1
            c_k, G_k = consjac(nlp, X[end])
            nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
            push!(KKT, norm([nab_xL_k; c_k]))
        end
    end
    Time = time() - Time
    if k == Max_Iter
        return [], [], [], Time, 0, 0
    else
        return X, Lam, KKT, Time, 1, 0
    end

end
