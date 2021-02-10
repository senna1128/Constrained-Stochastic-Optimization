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

cd("/Users/senna/course/Mihai/pro6/simu/code/AdapSQP")
###################################
### Select Problems        ########
###################################
#=
Problems1 = CUTEst.select(max_var=1000, min_con = 1, only_equ_con = true, objtype = "linear")
Problems2 = CUTEst.select(max_var=1000, min_con = 1, only_equ_con = true, objtype = "quadratic")
Problems3 = CUTEst.select(max_var=1000, min_con = 1, only_equ_con = true, objtype = "sum_of_squares")
Problems4 = CUTEst.select(max_var=1000, min_con = 1, only_equ_con = true, objtype = "other")
Problems = [Problems1; Problems2; Problems3; Problems4]
Prob = []
for prob in Problems
    nlp = CUTEstModel(prob)
    if length(nlp.meta.ifree) == nlp.meta.nvar && isempty(nlp.meta.nnet)
        if isempty(nlp.meta.jfree) && isempty(nlp.meta.jinf) && nlp.meta.minimize
            if length(nlp.meta.jfix) == nlp.meta.ncon
                push!(Prob, prob)
            end
        end
    end
    finalize(nlp)
end

# write problems in a file
writedlm(string(pwd(),"/Parameter/problems.txt"), Prob, ", ")
=#
## Exculde SPIN2OP, BT7 and ELEC

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
include("AdapMain.jl")


#######################################
#########  run main file    ###########
#######################################
function main()
    Random.seed!(2020)
    ## run nonadaptive SQP
    include("../Parameter/Param.jl")
    AdapR = AdapMain(Adap, Prob)
    if Adap.verbose
        write_matfile("../Solution/AdapSQP.mat"; AdapR)
    end
end

main()
