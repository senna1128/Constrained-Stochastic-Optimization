include("AdapSQP.jl")
struct AdapResult
    XStep::Array
    LamStep::Array
    KKTStep::Array
    Count_G_Step::Array
    Count_F_Step::Array
    alpha_Step::Array
    TimeStep::Array
end

## Implement Adaptive SQP for whole problem set
# Adap: parameters of adaptive algorithm
# Prob: problem name set

function AdapMain(Adap, Prob)
    Verbose = Adap.verbose
    Max_Iter = Adap.MaxIter
    EPS_Step = Adap.EPS_Step
    EPS_Res = Adap.EPS_Res
    nu = Adap.nu
    mu = Adap.mu
    epsilon = Adap.epsilon
    TotalRep = Adap.Rep
    beta = Adap.beta
    rho = Adap.rho
    alpha_max = Adap.alpha_max
    kap_grad = Adap.kap_grad
    kap_f = Adap.kap_f
    p_grad = Adap.p_grad
    p_f = Adap.p_f
    C_grad = Adap.C_grad
    Sigma = Adap.Sigma
    LenC_grad = length(C_grad)
    LenSigma = length(Sigma)

    AdapR = Array{AdapResult}(undef,length(Prob))

    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        nlp = CUTEstModel(Prob[Idprob])

        # define results vectors
        XStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        LamStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        KKTStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        Count_G_Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        Count_F_Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        alpha_Step = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))
        TimeStep = reshape([[] for i = 1:LenSigma*LenC_grad],(LenC_grad,LenSigma))

        # go over all cases
        i = 1
        while i <= LenC_grad
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("AdapSQP","-",Idprob,"-",i,"-",j,"-",rep)
                    X,Lam,KKT,Count_G,Count_F,Alpha,Time,IdCon,IdSing = AdapSQP(nlp,Sigma[j],Max_Iter,EPS_Step,EPS_Res,nu,mu,epsilon,beta,rho,alpha_max,kap_grad,kap_f,p_grad,p_f,C_grad[i])

                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(XStep[i,j], X)
                        push!(LamStep[i,j], Lam)
                        push!(KKTStep[i,j], KKT)
                        push!(Count_G_Step[i,j], Count_G)
                        push!(Count_F_Step[i,j], Count_F)
                        push!(alpha_Step[i,j], Alpha)
                        push!(TimeStep[i,j], Time)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end
        AdapR[Idprob] = AdapResult(XStep,LamStep,KKTStep,Count_G_Step,Count_F_Step,alpha_Step,TimeStep)
        finalize(nlp)
    end
    return AdapR
end
