#=
Pkg.add("NLPModels")
Pkg.add("JuMP")
Pkg.add("LinearOperators")
Pkg.add("OptimizationProblems")
Pkg.add("MathProgBase")
Pkg.add("ForwardDiff")
Pkg.add("CUTEst")
Pkg.add("NLPModelsJuMP")
Pkg.add("LinearAlgebra")
Pkg.add("Distributed")
Pkg.add("Ipopt")
Pkg.add("DataFrames")
Pkg.add("PyPlot")
Pkg.add("MATLAB")
Pkg.add("Glob")
=#

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

cd("/.../BerahasSQP")

######################################
######  Load problems    #############
######################################
Prob = readdlm(string(pwd(),"/../Parameter/problems.txt"))

# define parameter module
module Parameter
    struct NonAdapParams
        verbose                            # Do we create dump dir?
        NoAdapCAlpha::Array{Float64}       # Nonadaptive constant stepsize
        NoAdapDAlpha::Array{Float64}       # Nonadaptive decay stepsize 1/(K^p) with 0.5<p<1
        MaxIter::Int                       # Maximum Iteration
        Rep::Int                           # Number of Independent runs
        EPS::Float64                       # minimum of difference
        Sigma::Array{Float64}              # variance of gradient
    end

    struct BerahasParams
        verbose                            # Do we create dump dir?
        MaxIter::Int                       # Maximum Iteration
        EPS::Float64                       # minimum of difference
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

    struct AdapParams
        verbose                            # Do we create dump dir?
        MaxIter::Int                       # Maximum Iteration
        EPS::Float64                       # minimum of difference
        Rep::Int                           # Number of Independent runs
        nu::Float64                        # nu
        mu::Float64                        # mu
        epsilon::Float64                   # epsilon
        beta::Float64                      # beta
        rho::Float64                       # rho
        alpha_max::Float64                 # maximum of stepsize
        kap_grad::Float64                  # kappa of gradient
        kap_f::Float64                     # kappa of f
        p_grad::Float64                    # prob of gradient
        p_f::Float64                       # prob of f
        C_grad::Float64                    # constant of gradient
        Sigma::Array{Float64}              # variance of gradient
    end

end

using Main.Parameter
include("BerahasMain.jl")


#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2020)
    ## run nonadaptive SQP
    include("../Parameter/Param.jl")
    BerahasR = BerahasMain(Berahas, Prob)
    if Berahas.verbose
        BerahasR1 = BerahasR[1:20]
        write_matfile("../Solution/BerahasSQP1.mat"; BerahasR1)
        BerahasR2 = BerahasR[21:end]
        write_matfile("../Solution/BerahasSQP2.mat"; BerahasR2)
#        write_matfile("../Solution/BerahasSQP.mat"; BerahasR)
    end

end

main()
