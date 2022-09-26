using Distributions
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using Random
using DataStructures

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

function ps4_main()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # estimate a multinomial logit, panel form of the same data from PS3. 
    # I will use Dr. Ransom's solution for multinomial logit from PS3.
    # use automatic differentiation and compute standard errors. 

    function mlogit_with_Z(theta, X, Z, y)
        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println("logit estimates with Z")
    println([theta_hat_mle_ad theta_hat_mle_ad_se])

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    # 
    # Now γ = 1.3075, which is a positive number. This makes more sense since we expect  
    # utility to move positively with change in log wages, compared to the log wages
    # for the job category "Other". 
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # a) Approximate the integral by quadrature

    include("lgwt.jl")

    d = Normal(0,1)
    nodes, weights = lgwt(7, -4,4)
    println(sum(weights.*pdf.(d,nodes))) # 1.0044, about 1
    println(sum(weights.*nodes.*pdf.(d,nodes))) # 2.645e-17, very very close to zero

    # b) use quadrature to compute three integrals

    sigma = 2
    d2 = Normal(0,sigma)
    nodes, weights = lgwt(7, -5*sigma,5*sigma)
    println(sum(weights.*(nodes.^2).*pdf.(d2,nodes))) # 3.2655 

    # same integral but with 10 quadrature points

    nodes, weights =lgwt(10, -5*sigma, 5*sigma)
    println(sum(weights.*(nodes.^2).*pdf.(d2, nodes))) # 4.03898, closer to 4

    # As the number of quadrature points increases, the approximation is closer to 4, which is the
    # true variance of the normal distribution. 

    # c) Monte Carlo method

    function monte_carlo(D::Integer, a::Real=-1, b::Real=1)
        # returns the approximation of an integral by Monte Carlo simulation
    
        # set seed for replicability
        Random.seed!(1234)

        X_vector = zeros(D)

        unif_d = Uniform(a,b)
        for i in 1:D
            X_i = rand(unif_d)
            X_vector[i] = X_i
        end

        weights = (b-a)/D

        return X_vector, weights
    end

    D = 1_000_000

    x, weight = monte_carlo(D, -5*sigma, 5*sigma)


    println(sum(weight.*(x.^2).*pdf.(d2, x))) # 3.99747, quite close to σ^2=4

    println(sum(weight.*x.*pdf.(d2, x)))      # 0.0002595, near the mean = 0

    println(sum(weight.*pdf.(d2, x)))         # 1.00113, quite close to 1

    # try same as above but with different value of D, now D = 100

    D2 = 100

    x, weight = monte_carlo(D2, -5*sigma, 5*sigma)

    println(sum(weight.*(x.^2).*pdf.(d2, x))) # 4.085, still close to 4

    println(sum(weight.*x.*pdf.(d2, x)))      # 0.1189, not as close to the mean of 0

    println(sum(weight.*pdf.(d2, x)))         # 1.179, not as close to 1

    # overall, with a lower D, the approximation is not as near the true value compared to higher d

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # use quadrature to optimize the log likelihood function for the mixed logit with panel data
    # For both Question 4 and 5, I rewrote the code from Q1 to compute element wise..could not figure 
    # out for matrix notation

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # First I make 3D matrices. To make panel data, where I input 0's for all the X values if the date is missing for that row
    # dims=1 is the number of observations for that year
    # dims=2 is the number of explanatory variables
    # dims=3 is the number of years (68-88 for this specific dataset)

    year = df.year
    c = counter(year)
    rep = maximum(values(c))
    beg_yr = minimum(keys(c))
    end_yr = maximum(keys(c))

    X_matrix = hcat(year, X)

    function make_3d_array(X_matrix)
        matrix_3d = reshape(zeros(rep*size(X_matrix, 2)), (rep, size(X_matrix, 2)))
        for y in beg_yr:end_yr
            X_yr = X_matrix[X_matrix[:,1] .== y, :]

            add_rows = rep - size(X_yr, 1)
            miss_rows = repeat([0], add_rows, 3)
            miss_yr = repeat([y], add_rows, 1)
            miss_total = vcat(X_yr, hcat(miss_yr, miss_rows))
            matrix_3d = cat(matrix_3d, miss_total, dims=3)
        end
        return matrix_3d[:,Not(1),Not(1)]
    end

    X_3d = make_3d_array(X_matrix)
    Z_year = hcat(year, Z)
    Z_3d = make_3d_array(Z_year)

    y_year = hcat(year, y)
    y_3d = make_3d_array(y_year)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function mlogit_with_Z_quad(theta, X, Z, y)

        # parameters to estimate: alpha, gamma
    
        alpha = theta[begin:end-2]
        mu = theta[end-1]
        sigma = theta[end]
    
        dist_xi = Normal(mu, sigma)
        nodes, weights = lgwt(7, -5*sigma, 5*sigma)
        R = length(nodes)
    
        N = size(X, 1)                      # 2249
        K = size(X, 2)                      # 3
        J = size(Z, 2)                      # 8
        T = size(X, 3)                      # 21
    
        bigY = zeros(N, J, T)
    
        for t in 1:21
            for j in 1:8
                bigY[:,j, t] = y[:,:,t].==j
            end
        end
    
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)] # 3 by 8
        num_el = zeros(J)
        dem_el = 0
    
        loglike = 0
        for i in 1:N
            loglike_T = 0
            for t in 1:T
                quad_prod = zeros(R)
                for r in 1:R
                    dem_el = 0
                    for j in 1:J
                        num_el[j] = exp(X[i,:,t]'*bigAlpha[:,j] + (Z[i,j,t].-Z[i,J,t])*nodes[r]) 
                    end
                    dem_el = sum(num_el)
                    for j in 1:J
                        quad_prod[r] = weights[r]*prod((num_el[j] / dem_el).^bigY[i,j,t])*pdf.(dist_xi, nodes[r])
                    end
                end
                quad_sum = sum(quad_prod)
                loglike_T += log(quad_sum)
            end
            loglike += loglike_T
        end
        return -loglike
    end
    
    #startvals_quad = vcat(theta_hat_mle_ad[begin:end-1], [1, 2])
    
    #td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X_3d, Z_3d, y_3d), startvals; autodiff = :forward)
    #theta_hat_optim_quad = optimize(td, startvals_quad, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    
    function mlogit_with_Z_montecarlo(theta, X, Z, y)
    
        # parameters to estimate: alpha, gamma, μ_γ, σ_γ
    
        alpha = theta[begin:end-2]
        D = theta[end-1]
        sigma = theta[end]
    
        nodes, weights = monte_carlo(D, -5*sigma, 5*sigma)
    
        R = length(nodes)
    
        N = size(X, 1)                      # 2249
        K = size(X, 2)                      # 3
        J = size(Z, 2)                      # 8
        T = size(X, 3)                      # 21
    
        bigY = zeros(N, J, T)
    
        for t in 1:21
            for j in 1:8
                bigY[:,j, t] = y_3d[:,:,t].==j
            end
        end
    
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)] # 3 by 8
        num_el = zeros(J)
        dem_el = 0
    
        loglike = 0
        for i in 1:N
            loglike_T = 0
            for t in 1:T
                quad_prod = zeros(R)
                for r in 1:R
                    dem_el = 0
                    for j in 1:J
                        num_el[j] = exp(X[i,:,t]'*bigAlpha[:,j] + (Z[i,j,t].-Z[i,J,t])*nodes[r]) 
                    end
                    dem_el = sum(num_el)
                    for j in 1:J
                        quad_prod[r] = weights*prod((num_el[j] / dem_el).^bigY[i,j,t])*pdf.(dist_xi, nodes[r])
                    end
                end
                quad_sum = sum(quad_prod)
                loglike_T += log(quad_sum)
            end
            loglike += loglike_T
        end
        return -loglike
    end

    return nothing

end