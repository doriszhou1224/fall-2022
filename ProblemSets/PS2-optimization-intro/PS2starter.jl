using Optim
using FreqTables
using Random
using Distributions
using DataFrames
using CSV
using HTTP

function main_PS2()

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
    startval = rand(1)   # random starting value
    result = optimize(minusf, startval, LBFGS())
    println(result.minimizer)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_hat_ols.minimizer)

    bols = inv(X'*X)*X'*y
    println(bols)
    df.white = df.race.==1
    df.coll = df.collgrad.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println(bols_lm)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # function to maximize logit likelihood for a choice set with two alternatives

    function logit(alpha, X, d)

        loglike = -sum(d.*X*alpha - broadcast(log, ones(size(df,1)).+ℯ.^(X*alpha))) # the negative log likelihood 

        return loglike
    end

    beta_loglike = optimize(b -> logit(b, X, y), rand(size(X, 2)), LBFGS(),
                                    Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_loglike.minimizer)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # see Lecture 3 slides for example

    # Check MLE in question 3 for the logit likelihood. 
    a_hat = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println(a_hat)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8,:occupation] .= 7
    df[df.occupation.==9,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, d)

        # function to return the log likelihood function for multinomial logistic regression
    
        N = size(X,1)
        K = size(X,2)
        J = size(d,2)
        loglike = 0
        for i in 1:N
            y_j = 0
            for j in 1:J
                k_value = 0
                for k in 1:K
                    k_value += X[i,k]*alpha[k,j]
                end
                y_j += d[i,j]*k_value
            end
            e_value = 0
            for j in 1:J
                k_value2 = 0
                for k in 1:K
                    k_value2 += X[i,k]*alpha[k,j]
                end
                e_value += ℯ^k_value2
            end
            loglike += y_j - log(1+e_value)
        end
        return -loglike
    end

    # make a matrix Y of the y's that is 2237 by 6
    y_list = []
    for d in y
        row_y = zeros(7)
        row_y[d] = 1
        push!(y_list, row_y)
    end
    Y = reduce(hcat, y_list)'[:, 1:end-1]

    beta_mloglike = optimize(b -> mlogit(b, X, Y), rand(Uniform(-1,1), size(X, 2), 6), LBFGS(),
                                    Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    println(beta_mloglike.minimizer)

    # compared to Stata, the variable order β̂  =  beta_mloglike.minimizer gives for each column j is:
    # β̂ _{1j} = constant, β̂ _{2j} = age, β̂ _{3j} = white, β̂ _{4j} = collgrad

end

main_PS2()