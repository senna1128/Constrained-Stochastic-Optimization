include("BerahasSQP.jl")
struct BerahasResult
    XC::Array
    LamC::Array
    KKTC::Array
    TimeC::Array
    XD::Array
    LamD::Array
    KKTD::Array
    TimeD::Array
end


## Implement BerahasSQP for whole problem set
# Berahas: parameters of Berahas algorithm
# Prob: problem name set

function BerahasMain(Berahas, Prob)
    Max_Iter = Berahas.MaxIter
    EPS_Step = Berahas.EPS_Step
    EPS_Res = Berahas.EPS_Res
    TotalRep = Berahas.Rep
    tau = Berahas.tau
    epsilon = Berahas.epsilon
    sigma = Berahas.sigma
    xi = Berahas.xi
    theta = Berahas.theta
    VarSigma = Berahas.VarSigma
    StepCSet = Berahas.CBeta
    StepDSet = Berahas.DBeta
    LenCStep = length(StepCSet)
    LenDStep = length(StepDSet)
    LenVarSigma = length(VarSigma)

    BerahasR = Array{BerahasResult}(undef, length(Prob))

    # Go over all Problems
    for Idprob = 1:length(Prob)
        # load problem
        nlp = CUTEstModel(Prob[Idprob])

        # define results vector for constant stepsize
        XCStep = reshape([[] for i=1:LenCStep for j=1:LenVarSigma],LenCStep,:)
        LamCStep = reshape([[] for i=1:LenCStep for j=1:LenVarSigma],LenCStep,:)
        KKTCStep = reshape([[] for i=1:LenCStep for j=1:LenVarSigma],LenCStep,:)
        TimeCStep = reshape([[] for i=1:LenCStep for j=1:LenVarSigma],LenCStep,:)

        # go over constant stepsize
        i = 1
        while i <= LenCStep
            j = 1
            while j <= LenVarSigma
                rep = 1
                while rep <= TotalRep
                    println("Berahas ConstStep","-",Idprob,"-",i,"-",j,"-",rep)
                    X, Lam, KKT, Time, IdCon, IdSing = BerahasSQP(nlp,StepCSet[i],VarSigma[j],Max_Iter,EPS_Step,EPS_Res,tau,epsilon,sigma,xi,theta,1)
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

        # define results vector for decay stepsize
        XDStep = reshape([[] for i=1:LenDStep for j=1:LenVarSigma],LenDStep,:)
        LamDStep = reshape([[] for i=1:LenDStep for j=1:LenVarSigma],LenDStep,:)
        KKTDStep = reshape([[] for i=1:LenDStep for j=1:LenVarSigma],LenDStep,:)
        TimeDStep = reshape([[] for i=1:LenDStep for j=1:LenVarSigma],LenDStep,:)

        # go over constant stepsize
        i = 1
        while i <= LenDStep
            j = 1
            while j <= LenVarSigma
                rep = 1
                while rep <= TotalRep
                    println("Berahas DecayStep","-",Idprob,"-",i,"-",j,"-",rep)
                    X, Lam, KKT, Time, IdCon, IdSing = BerahasSQP(nlp,StepDSet[i],VarSigma[j],Max_Iter,EPS_Step,EPS_Res,tau,epsilon,sigma,xi,theta,0)
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
        BerahasR[Idprob] = BerahasResult(XCStep, LamCStep, KKTCStep, TimeCStep, XDStep, LamDStep, KKTDStep, TimeDStep)
        finalize(nlp)
    end
    return BerahasR
end
