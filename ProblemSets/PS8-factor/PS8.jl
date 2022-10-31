using Random
using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames
using CSV
using GLM
using FreqTables
using MultivariateStats

# import Pkg; Pkg.add("MultivariateStats")

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# load in the dataset

nlsy = CSV.read("ProblemSets/PS8-factor/nlsy.csv", DataFrame)

# estimate a linear regression model
nlsy_lm = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), nlsy)
println(nlsy_lm)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# compute the correlation among the six asvab variables

asvabMat = Matrix(nlsy[["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]])

cor_asvab = cor(asvabMat)
println(cor_asvab)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# regression with the six asvab variables
# from Question 2, the variables are quite correlated, some as high as 0.75,
# so would be problematic to directly put in the regression. 

asvab_lm = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), nlsy)
println(asvab_lm)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

M = fit(PCA, asvabMat'; maxoutdim=1)

asvabPCA = MultivariateStats.transform(M, asvabMat')
asvabPCA = asvabPCA'
nlsy.asvabPCA = asvabPCA

# regression
PCA_lm = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), nlsy)
println(PCA_lm)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Use FactorAnalysis instead of PCA

M_FA= fit(FactorAnalysis, asvabMat'; maxoutdim=1)

asvabFA = MultivariateStats.transform(M_FA, asvabMat')
asvabFA = asvabFA'
nlsy.asvabFA = asvabFA

# regression
FA_lm = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFA), nlsy)
println(FA_lm)


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Estimate the full measurement system with MLE

include("lgwt.jl")

# make X^m Matrix
Xm = [ones(size(nlsy,1),1) nlsy.black .==1 nlsy.hispanic.==1 nlsy.female.==1]
y = nlsy.logwage
X = [zeros(size(nlsy, 1)) (Matrix(nlsy[["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr"]]))]

function mle_quad(y, X, Xm, M, theta, R)
    # X is N by 7 matrix (with intercept)
    # Xm is N by 4 matrix
    # M is N by 6 matrix
    # R is the number of quadrature points (probably 7)

    N = size(Xm, 1)
    K = size(Xm, 2)
    J = size(M, 2)

    alpha = theta[1:K*(J)] # 24
    beta = theta[K*(J)+1:K*(J)+1+6] # 7
    gamma = theta[K*(J)+8:K*(J)+8+J] # 6
    delta = theta[K*(J)+8+J+1:K*(J)+8+J+1] #1
    #@show size(delta)
    sigma_j = exp.(theta[end-6:end-1]) # 6
    sigma_w = exp(theta[end]) # 1

    bigAlpha = reshape(alpha,K,J) # 4 by 6

    nodes, weights = lgwt(R,-4,4)

    T = promote_type(eltype(X),eltype(theta))
    out = zeros(T, N)

    for r in R
        L_result = zeros(N, 2)
        L_first = zeros(N, J)
        num   = zeros(T,N,J)
            for j=1:J
                num[:,j] = M[:,j] .- Xm*bigAlpha[:,j] .-gamma[j]*nodes[r] # N by 1 vector
                L_first[:,j] .= pdf(Normal(0, 1), num[:,j]./sigma_j[j])./sigma_j[j] # N by 1 vector
            end
            L_first_prod = prod.(eachrow(L_first))
            L_result[:,1] = L_first_prod
            
            L_result[:,2] = pdf(Normal(0, 1), (y .- X*beta .- delta*nodes[r])./sigma_w)./sigma_w # N by 1 vector

        for i in 1:N
            out[i] = weights[r].*L_result[i].*pdf(Normal(0,1), nodes[r])
        end
            
        return -sum(log.(out))
    end
end

startvals = zeros(45)
theta_hat_quad = optimize(theta -> mle_quad(y, X, Xm, asvabMat, theta, 7), startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
theta_hat_quad_ad = theta_hat_quad.minimizer


