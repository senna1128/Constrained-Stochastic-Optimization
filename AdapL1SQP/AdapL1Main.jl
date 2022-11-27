include("AdapL1SQP.jl")
struct AdapL1Result
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

function AdapL1Main(AdapL1, Prob)
    Verbose = AdapL1.verbose
    Max_Iter = AdapL1.MaxIter
    EPS_Step = AdapL1.EPS_Step
    EPS_Res = AdapL1.EPS_Res
    mu = AdapL1.mu
    epsilon = AdapL1.epsilon
    TotalRep = AdapL1.Rep
    beta = AdapL1.beta
    rho = AdapL1.rho
    alpha_max = AdapL1.alpha_max
    kap_grad = AdapL1.kap_grad
    kap_f = AdapL1.kap_f
    p_grad = AdapL1.p_grad
    p_f = AdapL1.p_f
    C_grad = AdapL1.C_grad
    Sigma = AdapL1.Sigma
    LenC_grad = length(C_grad)
    LenSigma = length(Sigma)

    AdapL1R = Array{AdapL1Result}(undef,length(Prob))

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
                    println("AdapL1SQP","-",Idprob,"-",i,"-",j,"-",rep)
                    X,Lam,KKT,Count_G,Count_F,Alpha,Time,IdCon,IdSing = AdapL1SQP(nlp,Sigma[j],Max_Iter,EPS_Step,EPS_Res,mu,epsilon,beta,rho,alpha_max,kap_grad,kap_f,p_grad,p_f,C_grad[i])

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
        AdapL1R[Idprob] = AdapL1Result(XStep,LamStep,KKTStep,Count_G_Step,Count_F_Step,alpha_Step,TimeStep)
        finalize(nlp)
    end
    return AdapL1R
end
