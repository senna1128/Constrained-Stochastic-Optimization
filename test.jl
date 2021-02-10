
Pkg.add("NLPModels")
Pkg.add("JuMP")


using NLPModels



f(x) = (x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2
x0 = [-1.2; 1.0]


adnlp = ADNLPModel(f, x0)


obj(adnlp, adnlp.meta.x0)
grad(adnlp, adnlp.meta.x0)
hess(adnlp, adnlp.meta.x0)
objgrad(adnlp, adnlp.meta.x0)
hprod(adnlp, adnlp.meta.x0, ones(2))
H = hess_op(adnlp, adnlp.meta.x0)
H * ones(2)

Pkg.add("LinearOperators")

using LinearOperators

n = 100
A = rand(n, n)
B = rand(n, n)
opA = LinearOperator(A)
opB = LinearOperator(B)
v = rand(n)



A = rand(5,5)
opA = LinearOperator(A)
A[:,1] * 3 # Vector
A[:,1] * [3] # Vector




using JuMP
using OptimizationProblems

using MathProgBase
using ForwardDiff

jmp = Model()
@variable(jmp, x[i=1:2], start=(x0[i])) # x0 from before
@NLobjective(jmp, Min, (x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2)
mpbnlp = MathProgNLPModel(jmp)


using NLPModels, MathProgBase, JuMP
m = Model()
@variable(m, x[1:4])
@NLobjective(m, Min, sum(x[i]^4 for i=1:4))
nlp = MathProgNLPModel(m)
x0 = [1.0; 0.5; 0.25; 0.125]
grad(nlp, x0)



using CUTEst

nlp = CUTEstModel("ROSENBR")





function newton(nlp :: AbstractNLPModel)
  x = nlp.meta.x0
  fx = obj(nlp, x)
  gx = grad(nlp, x)
  ngx = norm(gx)
  while ngx > 1e-6
    Hx = hess(nlp, x)
    d = -gx
    try
      G = cholesky(Hermitian(Hx, :L)) # Make Cholesky work on lower-only matrix.
      d = G\-gx # G'\(G\-gx) if G = cholesky(Hermitian(Hx, :L)).L
    catch e
      if !isa(e, LinearAlgebra.PosDefException); rethrow(e); end
    end
    t = 1.0
    xt = x + t * d
    ft = obj(nlp, xt)
    while ft > fx + 0.5 * t * dot(gx, d)
      t *= 0.5
      xt = x + t * d
      ft = obj(nlp, xt)
    end
    x = xt
    fx = ft
    gx = grad(nlp, x)
    ngx = norm(gx)
  end
  return x, fx, ngx
end


X, Lam, KKT, Count, Time, IdCon, IdSing = AdapSQP(nlp,1e-8,Max_Iter,EPS,nu,mu,epsilon,beta,rho,alpha_max,kap_grad,kap_f,p_grad,p_f,C_grad)


[[AdapR[4].KKTStep[i][j][end] for j = 1:5] for i = 1:5]


X, Lam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp, 0.6, 0.1, Max_Iter, EPS, 1)

[[BerahasR[1].KKTC[i,2][1][end] for i = 1:4]]

using JuMP
jmp = Model(with_optimizer(Ipopt.Optimizer))
@variable(jmp, x[1:2]) # x0 from before
@NLobjective(jmp, Min, (x[1] - 1)^2 + 100*(x[2] - x[1])^2)
@NLconstraint(jmp, eqconi[1], x[1]- x[2] == 0)
set_start_value(x[1],0);
set_start_value(x[2],0);
nlp = MathOptNLPModel(jmp)
optimize!(jmp)
xout = JuMP.value.(x)

X, Lam, KKT, Time, IdCon, IdSing = BerahasSQP(nlp,Step,1e-6,1000000,1e-4,tau,epsilon,sigma,xi,theta,1)


X, Lam, KKT, Time, IdCon, IdSing = BerahasSQP(nlp,0.1,1e-4,100000,1e-4,1,1e-6,0.5,1,10,1)

BerahasR[1].KKTC[i, j][1][end] for i = 1:4 for j = 1:5

eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
c_k, G_k = consjac(nlp, X[end])



include("NonAdapSQP.jl")


X, Lam, KKT, Count, Time, IdCon, IdSing = AdapSQP(nlp,1e-8,100000,1e-4,0.01,1,1,0.3,1.2,1.5,1,0.04,0.9,0.9,2)



X, Lam, KKT, Time, IdCon, IdSing = BerahasSQP(nlp, 1.0, 1e-8, 1000000, EPS, 1, 1e-6, 0.5,1,10,1)

X, Lam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp, 1, 1e-8, Max_Iter, EPS, 1)

X, Lam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp, 1.0, 1e-4, 10000, EPS, 1.0)

A, B, C, D, E, F, G = NonAdapSQP(nlp, 1.0, 1, 3, EPS, 1.0)

A = 10
B = 2

while true
    A *= B
    if B>10
        break
    else
        B*=10
    end
end


nx = nlp.meta.nvar
nlam = nlp.meta.ncon
# Initialize
eps, k, X, Lam = 1, 0, [nlp.meta.x0], [nlp.meta.y0]
c_k, G_k = cons(nlp, X[end]), jac(nlp, X[end])
nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end], zeros(nlam))
#    nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
nab_xL_k = nabf_k + G_k'Lam[end]
nab_x2L_k = hess(nlp, X[end], Lam[end])
#    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
KKT = [norm([nab_xL_k; c_k])]
CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
IdCon, IdSing = 1, 0

while min(eps, KKT[end]) > EPS && k < Max_Iter
    ## Obtain est of gradient of Lagrange
#        bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
#        new_bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
    bnab_xL_k = nab_xL_k
    new_bnab_xL_k = nab_xL_k
    ## Obtain est of Hessian of Lagrange
#        Delta = rand( Normal(0,sigma^(1/2)), nx, nx)
#        bnab_x2L_k = nab_x2L_k + (Delta + Delta')/2
    bnab_x2L_k = nab_x2L_k
    ## Compute M_k and T_k
    bT_k = zeros(nx, nlam)
    for i = 1:nlam
        bT_k[:, i] = (hess(nlp,X[end],convert(Array{Float64}, 1:nlam.==i))-nab2f_k)*new_bnab_xL_k
    end
    bM_k = bnab_x2L_k*G_k' + bT_k
    ## Compute Newton Direction
    try
#            FullH = hcat(vcat(bnab_x2L_k, G_k), vcat(G_k', zeros(nlam, nlam)))
        FullH = hcat(vcat(Diagonal(ones(nx)), G_k), vcat(G_k', zeros(nlam, nlam)))
        FullG = vcat(bnab_xL_k, c_k)
        NewDir = lu(FullH)\-FullG
        NewDir[nx+1:end] = lu(G_k*G_k')\-(G_k*bnab_xL_k + bM_k'NewDir[1:nx])
    catch
        IdSing = 1
    end
    if IdSing == 1
        println("abc")
    else
        Stepsize = IdConst*Step+(1-IdConst)/(k+1)^Step
        push!(X, X[end] + Stepsize*NewDir[1:nx])
        push!(Lam, Lam[end] + Stepsize*NewDir[nx+1:end])
        eps, k = norm(Stepsize * NewDir), k+1
        c_k, G_k = cons(nlp, X[end]), jac(nlp, X[end])
        nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end], zeros(nlam))
#            nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
        nab_xL_k = nabf_k + G_k'Lam[end]
        nab_x2L_k = hess(nlp, X[end], Lam[end])
#            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
        push!(KKT, norm([nab_xL_k; c_k]))
    end
end
Time = time() - Time
if k == Max_Iter
    println("def")
else
    println("success")
end










 newton(nlp)


finalize(nlp)
nlp = CUTEstModel("HIMMELBB")
x, fx, ngx = newton(nlp)
finalize(nlp)



nlp = CUTEstModel("BOX")
nlp.meta.nvar

function foo1()
  H = hess(nlp, nlp.meta.x0)
  v = ones(nlp.meta.nvar)
  return Hermitian(H, :L) * v
end

function foo2()
  H = hess_op(nlp, nlp.meta.x0)
  v = ones(nlp.meta.nvar)
  return H * v
end

@time w1 = foo1();
@time w2 = foo2();
norm(w1 - w2)
finalize(nlp)




using NLPModelsJuMP


using LinearAlgebra


newton(adnlp)
newton(mpbnlp)

using OptimizationProblems
using NLPModels

x, fx, ngx = newton(MathOptNLPModel( rosenbrock() ))
x, fx, ngx = newton(MathOptNLPModel( dixmaanj() ))
x, fx, ngx = newton(MathOptNLPModel( brownbs() ))


fetch_sif_problems()

using CUTEst

nlp = CUTEstModel("BYRDSPHR");
print(nlp);

fx = obj(nlp, nlp.meta.x0)
gx = grad(nlp, nlp.meta.x0)
Hx = hess(nlp, nlp.meta.x0)
finalize(nlp)

using Distributed


function decodemodel(name)
    finalize(CUTEstModel(name))
end
probs = ["AKIVA", "ALLINITU", "ARGLINA", "ARGLINB", "ARGLINC","ARGTRIGLS", "ARWHEAD"]
broadcast(decodemodel, probs)


addprocs(2)
@everywhere using CUTEst
@everywhere function evalmodel(name)
   nlp = CUTEstModel(name, decode = false)
   retval = obj(nlp, nlp.meta.x0)
   finalize(nlp)
   retval
end

fvals = pmap(evalmodel, probs)




set_mastsif()

for pro in probs
    evalmodel(pro)
end


addprocs(2)
@everywhere using CUTEst
@everywhere function opt(name, decode)
    nlp = CUTEstModel(name;decode=decode)
    retval = obj(nlp,nlp.meta.x0)
    #retval = nlp.meta.nvar
    finalize(nlp)
    retval
end

probs = sort(CUTEst.select(contype="unc",max_var=5))
map(x->opt(x,true), probs) # Ensure all the problems have been decoded
pmap(x->opt(x,false), probs)








function NonAdapSQP(nlp, Step, sigma, Max_Iter, EPS, IdConst)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon

    # Initialize
    eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
    c_k, G_k = cons(nlp, X[end]), jac(nlp, X[end])
    nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end], zeros(nlam))
#    nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], Lam[end])
#    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    KKT = [norm([nab_xL_k; c_k])]
    CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
    IdCon, IdSing = 1, 0

    # Time
    Time = time()
    while min(eps, KKT[end]) > EPS && k < Max_Iter
        ## Obtain est of gradient of Lagrange
        bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
        new_bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
        ## Obtain est of Hessian of Lagrange
        Delta = rand(Normal(0,sigma^(1/2)), nx, nx)
        bnab_x2L_k = Hermitian(nab_x2L_k, :L) + (Delta + Delta')/2
        ## Compute M_k and T_k
        bT_k = zeros(nx, nlam)
        for i = 1:nlam
            bT_k[:, i] = Hermitian(hess(nlp,X[end],convert(Array{Float64}, 1:nlam.==i)) - nab2f_k,:L) * new_bnab_xL_k
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
            c_k, G_k = cons(nlp, X[end]), jac(nlp, X[end])
            nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end], zeros(nlam))
#            nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = hess(nlp, X[end], Lam[end])
#            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
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



function NonAdapSQP(nlp, Step, sigma, Max_Iter, EPS, IdConst)
    nx = nlp.meta.nvar
    nlam = nlp.meta.ncon

    # Initialize
    eps, k, X, Lam, NewDir = 1, 0, [nlp.meta.x0], [nlp.meta.y0], zeros(nx+nlam)
    c_k, G_k = cons(nlp, X[end]), jac(nlp, X[end])
    nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end], zeros(nlam))
#    nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
    nab_xL_k = nabf_k + G_k'Lam[end]
    nab_x2L_k = hess(nlp, X[end], Lam[end])
#    nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
    KKT = [norm([nab_xL_k; c_k])]
    CovM = sigma*(Diagonal(ones(nx)) + ones(nx, nx))
    IdCon, IdSing = 1, 0

    # Time
    Time = time()
    while min(eps, KKT[end]) > EPS && k < Max_Iter
        ## Obtain est of gradient of Lagrange
        bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
        new_bnab_xL_k = rand(MvNormal(nab_xL_k, CovM))
        ## Obtain est of Hessian of Lagrange
        Delta = rand(Normal(0,sigma^(1/2)), nx, nx)
        bnab_x2L_k = Hermitian(nab_x2L_k, :L) + (Delta + Delta')/2
        ## Compute M_k and T_k
        bT_k = zeros(nx, nlam)
        for i = 1:nlam
            bT_k[:, i] = Hermitian(hess(nlp,X[end],convert(Array{Float64}, 1:nlam.==i)) - nab2f_k,:L) * new_bnab_xL_k
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
            c_k, G_k = cons(nlp, X[end]), jac(nlp, X[end])
            nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end], zeros(nlam))
#            nabf_k, nab2f_k = grad(nlp, X[end]), hess(nlp, X[end])
            nab_xL_k = nabf_k + G_k'Lam[end]
            nab_x2L_k = hess(nlp, X[end], Lam[end])
#            nab_x2L_k = hess(nlp, X[end], obj_weight=1.0, Lam[end])
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
