using Random
using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using SMM


function main_PS7()
    #:::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::

    # Estimate linear regression by GMM

    # Load in the data

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols_gmm(α, X, y)
        g = y .- X*α
        J = g'*I*g
        return J
    end
    α̂_optim = optimize(a -> ols_gmm(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
    println(α̂_optim.minimizer)

    df.white = df.race.==1
    df.coll = df.collgrad.==1
    bols_lm = lm(@formula(married ~ age + white + coll), df)
    println(bols_lm)

    # yes the coefficients agree

    #::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    #::::::::::::::::::::::::::::::::::::::::::::::::

    # data, code from main_PS2
    df = dropmissing(df, :occupation)
    df[df.occupation.==8,:occupation] .= 7
    df[df.occupation.==9,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation)

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    # a) Maximum likelihood

    # rerun from Q5, from PS2 solutions

    function mlogit(alpha, X, y)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum(bigY.*log.(P))
        
        return loglike
    end

    # alpha_rand = rand(6*size(X,2))
    alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    alpha_start = alpha_true.*rand(size(alpha_true))
    println(size(alpha_true))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)

    # b) GMM with the MLE estimates as starting values. 

    function mlogit_gmm(α, X, y)
        J = length(unique(y))
        N = length(y)

        y_matrix = zeros(N,J)
        for j=1:J
            y_matrix[:,j] = y.==j
        end
        # y_matrix = vec(y_matrix)
        # @show size(y_matrix)
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*α[:,j])
            dem .+= num[:,j]
        end
        
        # P = vec(exp.(X*α)./(1 .+ exp.(X*α)))

        P = num./repeat(dem,1,J)
        # @show size(P)
        #g = vec(y_matrix .- P)
        g = y_matrix[:]-P[:]
        # @show size(g)
        J = g'*I*g
        return J
    end
    mle_estimates = [reshape(alpha_hat_mle,size(X,2),length(unique(y))-1) zeros(size(X,2))]
    
    α̂_optim = optimize(a -> mlogit_gmm(a, X, y), mle_estimates, LBFGS(), Optim.Options(g_tol=1e-8, show_trace=true, iterations=100_000))
    println(α̂_optim.minimizer)

    # c) Random starting values

    Random.seed!(1234)                    

    gmm_random = rand(Uniform(-5,10), size(X, 2), length(unique(y)))
    rand_optim = optimize(a -> mlogit_gmm(a, X, y), gmm_random, LBFGS(), Optim.Options(g_tol=1e-8, show_trace=true, iterations=100_000))
    println(rand_optim.minimizer)

    # The estimates from part c) are not the similar to the estimates from part b) and converged much faster. So the objective function is not globally concave.

    #:::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::

    # Simulate dataset 
    # From juliahub: https://docs.juliahub.com/CalculusWithJulia/AZHbv/0.0.5/precalc/ranges.html

    function evenly_spaced(a, b, n)
        h = (b-a)/(n-1)
        collect(a:h:b)
    end

    function simulate_data(N, K, J)
        Random.seed!(1234)

        # a) make random X
        X = randn(N, K)

        # b) Set values of β
        β1=evenly_spaced(-1, 1, K*(J-1))
        β = [reshape(β1, K, J-1) zeros(K)]

        # c) Generate N × J matrix of probabilities
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*β[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)

        # d) Draw preference shocks ϵ from from U[0, 1] as N×1 vector
        ϵ = rand(Uniform(0,1), N, 1)

        # e) Generate y
        Y = zeros(N)
        for i in 1:N
            indicator = 0
            for j in 1:J
                temp_P = 0
                for k in j:J
                    temp_P += P[i,k]
                end
                if temp_P > ϵ[i]
                    indicator += 1
                else 
                    indicator += 0
                end
            end
           
            Y[i] = indicator
        end
       # return X, Y, β, P, ϵ

       return X, Y, β
    end

    # N = 10, K = 2, J = 4
    X, Y, beta = simulate_data(10, 2, 4)

    test_hat_optim = optimize(a -> mlogit(a, X, Y), vec(beta), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    test_hat_mle = test_hat_optim.minimizer
    println(test_hat_mle)

    #:::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4 
    #:::::::::::::::::::::::::::::::::::::::::::::::::

    # Use SMM.jl to run example code on slide #21

    # code from slide #21

    MA = SMM.parallelNormal() # Note: this line may take up to 5 minutes to execute
    dc = SMM.history(MA.chains[1])
    dc = dc[dc[:accepted].==true, :]
    println(describe(dc))

    #:::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::

    # Run SMM on data from Question 3

    function mlogit_smm(θ, X, y, D)
        K = size(X,2)
        N = size(y,1)
        J = length(unique(y))
        β = [reshape(θ, K, J)]

        y_matrix = zeros(N,J)
        for j=1:J
            y_matrix[:,j] = y.==j
        end
        y_matrix = vec(y_matrix)

        # N+1 moments in both model and data
        gmodel = zeros(N*J,D)
        # data moments are just the y vector itself
        # and the variance of the y vector
        gdata  = vec(y_matrix)
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####
        # This is critical!                   #
        Random.seed!(1234)                    #
        # You must always use the same ε draw #
        # for every guess of θ!               #
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
        # simulated model moments

        for d=1:D
            num = zeros(N,J)
            dem = zeros(N)
            for j=1:J
                num[:,j] = exp.(X*β[:,j])
                dem .+= num[:,j]
            end
            P = num./repeat(dem,1,J)

            ỹ = P             # since the Y's are generated from Question 3 with random ϵ
            gmodel[1:end,d] = vec(ỹ)
           # gmodel[  end  ,d] = var(ỹ)
        end
        # criterion function
        err = vec(gdata .- mean(gmodel; dims=2))
        # weighting matrix is the identity matrix
        # minimize weighted difference between data and moments
        J = err'*I*err
        return J
    end

    # try with true beta as starting values
    true_beta = vec(beta)   
    smm_optim = optimize(a -> mlogit_smm(a, X, Y, 1000), true_beta, LBFGS(), Optim.Options(g_tol=1e-8, show_trace=true, iterations=100_000))
    println(smm_optim.minimizer)

    # try with random beta as starting values
    Random.seed!(1234)
    rand_beta = rand(8)
    rsmm_optim = optimize(a -> mlogit_smm(a, X, Y, 1000), rand_beta, LBFGS(), Optim.Options(g_tol=1e-8, show_trace=true, iterations=100_000))
    println(rsmm_optim.minimizer)

   return nothing

end