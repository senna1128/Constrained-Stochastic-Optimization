include("AdapSQP.jl")
struct AdapResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    CountStep::Array
    TimeStep::Array
end

## Implement NonAdaptive SQP for whole problem set
# NonAdap: parameters of Nonadaptive algorithm
# Prob: problem name set

function AdapMain(Adap, Prob)
    Verbose = Adap.verbose
    Max_Iter = Adap.MaxIter
    EPS = Adap.EPS
    TotalRep = Adap.Rep
    nu = Adap.nu
    mu = Adap.mu
    epsilon = Adap.epsilon
    beta = Adap.beta
    rho = Adap.rho
    alpha_max = Adap.alpha_max
    kap_grad = Adap.kap_grad
    kap_f = Adap.kap_f
    p_grad = Adap.p_grad
    p_f = Adap.p_f
    C_grad = Adap.C_grad
    Sigma = Adap.Sigma
    LenSigma = length(Sigma)

    AdapR = Array{AdapResult}(undef,length(Prob))

    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        nlp = CUTEstModel(Prob[Idprob])

        # define results vector for constant stepsize
        XStep = [[] for i=1:LenSigma]
        LamStep = [[] for i=1:LenSigma]
        KKTStep = [[] for i=1:LenSigma]
        CountStep = [[] for i=1:LenSigma]
        TimeStep = [[] for i=1:LenSigma]

        # go over constant stepsize
        i = 1
        while i <= LenSigma
            rep = 1
            while rep <= TotalRep
                println("AdapSQP", Idprob, i, rep)
                X, Lam, KKT, Count, Time, IdCon, IdSing = AdapSQP(nlp,Sigma[i],Max_Iter,EPS,nu,mu,epsilon,beta,rho,alpha_max,kap_grad,kap_f,p_grad,p_f,C_grad)
                if IdSing == 1
                    break
                elseif IdCon == 0
                    rep += 1
                else
                    push!(XStep[i], X)
                    push!(LamStep[i], Lam)
                    push!(KKTStep[i], KKT)
                    push!(CountStep[i], Count)
                    push!(TimeStep[i], Time)
                    rep += 1
                end
            end
            i += 1
        end
        AdapR[Idprob] = AdapResult(XStep, LamStep, KKTStep, CountStep, TimeStep)
        finalize(nlp)
    end
    return AdapR
end
