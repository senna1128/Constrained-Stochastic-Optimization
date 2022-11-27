include("NonAdapSQP.jl")
struct NonAdapResult
    XC::Array
    LamC::Array
    KKTC::Array
    TimeC::Array
    XD::Array
    LamD::Array
    KKTD::Array
    TimeD::Array
end

## Implement NonAdaptive SQP for whole problem set
# NonAdap: parameters of Nonadaptive algorithm
# Prob: problem name set

function NonAdapMain(NonAdap, Prob)
    Verbose = NonAdap.verbose
    StepCSet = NonAdap.NoAdapCAlpha
    StepDSet = NonAdap.NoAdapDAlpha
    Max_Iter = NonAdap.MaxIter
    TotalRep = NonAdap.Rep
    EPS_Step = NonAdap.EPS_Step
    EPS_Res = NonAdap.EPS_Res
    Sigma = NonAdap.Sigma
    LenCStep = length(StepCSet)
    LenDStep = length(StepDSet)
    LenSigma = length(Sigma)

    NonAdapR = Array{NonAdapResult}(undef,length(Prob))

    ## Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        nlp = CUTEstModel(Prob[Idprob])

        # define results vector for constant stepsize
        XCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        LamCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        KKTCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)
        TimeCStep = reshape([[] for i=1:LenCStep for j=1:LenSigma], LenCStep,:)

        # go over constant stepsize
        i = 1
        while i <= LenCStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("NonAdapSQP ConstStep","-",Idprob,"-",i,"-",j,"-",rep)
                    X, Lam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp, StepCSet[i], Sigma[j], Max_Iter, EPS_Step,EPS_Res,1)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(XCStep[i, j], X)
                        push!(LamCStep[i, j], Lam)
                        push!(KKTCStep[i, j], KKT)
                        push!(TimeCStep[i, j], Time)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end


        # define results vector for constant stepsize
        XDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        LamDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        KKTDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)
        TimeDStep = reshape([[] for i=1:LenDStep for j=1:LenSigma], LenDStep,:)

        # go over decay stepsize
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenSigma
                rep = 1
                while rep <= TotalRep
                    println("NonAdapSQP DecayStep","-",Idprob,"-",i,"-",j,"-",rep)
                    X, Lam, KKT, Time, IdCon, IdSing = NonAdapSQP(nlp, StepDSet[i], Sigma[j], Max_Iter, EPS_Step,EPS_Res, 0)
                    if IdSing == 1
                        break
                    elseif IdCon == 0
                        rep += 1
                    else
                        push!(XDStep[i, j], X)
                        push!(LamDStep[i, j], Lam)
                        push!(KKTDStep[i, j], KKT)
                        push!(TimeDStep[i, j], Time)
                        rep += 1
                    end
                end
                j += 1
            end
            i += 1
        end
        NonAdapR[Idprob] = NonAdapResult(XCStep, LamCStep, KKTCStep, TimeCStep, XDStep, LamDStep, KKTDStep, TimeDStep)
        finalize(nlp)
    end
    return NonAdapR
end
