using CSV
using DataFrames
using Distributions
using HTTP
using Random
using GLM
using Optim

function PS3_main() 

    # parts of the assignment were done with Will Meyers

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # function to estimate multinomial logit
     
    y_list = []
    for d in y
        row_y = zeros(size(Z, 2))
        row_y[d] = 1
        push!(y_list, row_y)
    end
    Y = reduce(hcat, y_list)'[:, 1:end-1]
    
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
    #result = mlogit(alpha_test, X_const, Y, Z)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #= From Question 1, \gamma = -0.0942. This means that log odds for the wage of seven choice sets
    is -0.0942 times greater than the wage for 'Other'. 
    =# 

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function nlogit(coef_vector, Z, X, y)

        # Number of dependent variables in X
        K = size(X,2)
    
        # Number of choice variables (occupations)
        J = length(unique(y))
    
        # Number of observations
        N = size(X, 1)
    
        beta_WC = coef_vector[1:3]
        beta_BC = coef_vector[4:6]
        lambda_WC = coef_vector[7]
        lambda_BC = coef_vector[8]
        gamma = coef_vector[9]

        Z_indexed = Z .- Z[:,K]
    
        # Create numerator for each occupation
        num1 = exp.((X * beta_WC .+ gamma * Z_indexed[:,1]) ./ lambda_WC)
        num2 = exp.((X * beta_WC .+ gamma * Z_indexed[:,2]) ./ lambda_WC)
        num3 = exp.((X * beta_WC .+ gamma * Z_indexed[:,3]) ./ lambda_WC)
    
    
        num4 = exp.((X * beta_BC .+ gamma * Z_indexed[:,4]) ./ lambda_BC)
        num5 = exp.((X * beta_BC .+ gamma * Z_indexed[:,5]) ./ lambda_BC)
        num6 = exp.((X * beta_BC .+ gamma * Z_indexed[:,6]) ./ lambda_BC)
        num7 = exp.((X * beta_BC .+ gamma * Z_indexed[:,7]) ./ lambda_BC)
    
    
        num8 = exp.(X * zeros(3) .+ gamma * Z_indexed[:,8])
    
        # Create denominator
    
        den_WC1 = ((num1) .+ (num2) .+ (num3))
        den_WC1
    
        # println(den_WC1)
    
        den_BC1 = ((num4) .+ (num5) .+ (num6) .+ (num7))
        den2 = 1 .+ den_WC1.^lambda_WC .+ den_BC1.^lambda_BC
    
        P = zeros(N, J)
        for j in 1:J
            if j <= 3
                num_p = exp.((X * beta_WC .+ gamma * Z_indexed[:,j])./ lambda_WC) .* den_WC1.^(lambda_WC-1)
                den_p = den2
    
                P[:,j] .= num_p ./ den_p
            end
    
            if 4 <= j <= 7
                num_p = exp.((X * beta_BC .+ gamma * Z_indexed[:,j])./ lambda_BC) .* den_BC1.^(lambda_BC-1)
                den_p = den2
    
                P[:,j] .= num_p ./ den_p
            end
    
            if j == 8
    
                num_p = num8
                den_p = den2
    
                # println(num_p ./ den_p)
                P[:,j] .= num_p ./ den_p
    
            end
        end
    
        # Create choice dummy variable matrix used in log likelihood
        D = zeros(N,J)
            for j=1:J
                D[:,j] = y.==j
            end
    
        #loglike = sum((D[:,1] .* log.(prob1)) .+ (D[:,2] .* log.(prob2)) .+ (D[:,3] .* log.(prob3))  .+ (D[:,4] .* log.(prob4))  .+ (D[:,5] .* log.(prob5))  .+ (D[:,6] .* log.(prob6)) .+ (D[:,7] .* log.(prob7)) .+ (D[:,8] .* log.(prob8)))
    
        loglike = sum(D .* broadcast(log, P))
        return -loglike
    end
    
    test_alpha = rand(Uniform(0, 1), 9)
    beta_hat_loglike = optimize(alpha -> nlogit(alpha, Z, X, y), test_alpha, BFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(beta_hat_loglike.minimizer)

end



