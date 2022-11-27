

## Load packages
using NLPModels
using JuMP
using LinearOperators
using OptimizationProblems
using MathProgBase
using ForwardDiff
using CUTEst
using NLPModelsJuMP
using LinearAlgebra
using Distributed
using Ipopt
using DataFrames
using PyPlot
using MATLAB
using Glob
using DelimitedFiles
using Random
using Distributions

cd("/.../AdapSQP")
Prob = readdlm(string(pwd(),"/../Parameter/problems.txt"))

# define parameter module
module Parameter
    # Parameters of adaptiveSQP with augmented Lagrangian
    struct AdapParams
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        # adaptive parameters
        nu::Float64                        # nu
        mu::Float64                        # mu
        epsilon::Float64                   # epsilon
        # fixed parameters
        Rep::Int                           # Number of Independent runs
        beta::Float64                      # beta
        rho::Float64                       # rho
        alpha_max::Float64                 # maximum of stepsize
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of gradient
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        # test parameters
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

    struct AdapL1Params
        verbose                            # Do we create dump dir?
        # stopping parameters
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        # adaptive parameters
        mu::Float64                        # mu
        epsilon::Float64                   # epsilon
        # fixed parameters
        Rep::Int                           # Number of Independent runs
        beta::Float64                      # beta
        rho::Float64                       # rho
        alpha_max::Float64                 # maximum of stepsize
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of gradient
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        # test parameters
        C_grad::Array{Float64}             # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end


    struct NonAdapParams
        verbose                            # Do we create dump dir?
        NoAdapCAlpha::Array{Float64}       # Nonadaptive constant stepsize
        NoAdapDAlpha::Array{Float64}       # Nonadaptive decay stepsize 1/(K^p) with 0.5<p<1
        MaxIter::Int                       # Maximum Iteration
        Rep::Int                           # Number of Independent runs
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        Sigma::Array{Float64}              # variance of gradient
    end

    struct BerahasParams
        verbose                            # Do we create dump dir?
        MaxIter::Int                       # Maximum Iteration
        EPS_Step::Float64                  # minimum of difference
        EPS_Res::Float64                   # minimum of difference
        Rep::Int                           # Number of Independent runs
        tau::Float64                       # tau
        epsilon::Float64                   # epsilon
        sigma::Float64                     # sigma
        xi::Float64                        # xi
        theta::Float64                     # theta
        CBeta::Array{Float64}              # constant stepsize
        DBeta::Array{Float64}              # decay stepsize
        VarSigma::Array{Float64}           # variance of gradient
    end

end



using Main.Parameter
include("AdapMain.jl")


#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2021)
    ## run nonadaptive SQP
    include("../Parameter/Param.jl")
    AdapR = AdapMain(Adap, Prob)

    if Adap.verbose
        NumProb = 10
        decom = convert(Int64, floor(length(AdapR)/NumProb))
        for i=1:decom
            path = string("../Solution/AdapSQP", i, ".mat")
            Result = AdapR[(i-1)*NumProb+1:i*NumProb]
            write_matfile(path; Result)
        end
        path = string("../Solution/AdapSQP", decom+1, ".mat")
        Result = AdapR[decom*NumProb+1:end]
        write_matfile(path; Result)
    end
end

main()
