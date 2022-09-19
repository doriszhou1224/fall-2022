using CSV
using DataFrames
using Distributions
using HTTP
using Random
using GLM
using Optim

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

function mlogit(alpha, X, Z, y)
    #beta = alpha[begin:end-1, begin:end] # 3 by 7
    #gamma = alpha[end:end, begin:end][1] # a scalar 

    N = size(X, 1)
    J = size(Z, 2)
    K = size(X, 2)

    diffed_Z = Z .- Z[:,J]
    #diffed_Z = diffed_Z[begin:end, begin:end-1]
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    beta = hcat([reshape(alpha[1:21],K,J-1)][1], zeros(K))
    #println(beta[1])
    gamma = alpha[end]
    #println(gamma)

    num = zeros(N, J)
    for j in 1:J
        
        num[:,j] = ℯ.^(X*beta[:,j] .+ gamma*diffed_Z[:,j])
    end
    dem = sum(num, dims=2)

    P = zeros(N, J)
    for j in 1:J
        if j < 8
            num1 = ℯ.^(X*beta[:,j].+gamma*diffed_Z[:,j])
            P[:,j] .= num1./(dem)
        end
        if j == 8
            num1 = 1
            P[:,j] .= num1./(dem)
        end
    end
#    @show P

    loglike = sum(bigY.*broadcast(log, P))

    return -loglike
end

test_alpha = rand(Uniform(0,1), 22)
alpha_hat_optim = optimize(alpha -> mlogit(alpha, X, Z, y), test_alpha, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim.minimizer)