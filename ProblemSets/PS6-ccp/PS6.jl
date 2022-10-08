using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# read in function to create state transitions for dynamic model
include("create_grids.jl")

function main_PS6()

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # load in the data and make the data long, recycling the code from PS5

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame) 


    # create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1)) # 1000 by 64

    #---------------------------------------------------
    # reshape from wide to long (must do this twice be-
    # cause DataFrames.stack() requires doing it one 
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id])) # 20000 by 3. 20*1000 = 20000
    rename!(dfx_long, :value => :Odometer)           # renaming column value to Odometer
    dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time]) 

    # df_long is 20000x6

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Estimate a flexible logit model

    # Make the predictor variables
    mileage = df_long[!, :Odometer]
    mileage2 = mileage.^2
    RouteUsage = df_long[!, :RouteUsage]
    RouteUsage2 = RouteUsage.^2
    Branded = df_long[!, :Branded]
    TimePeriod = df_long[!, :time]
    TimePeriod2 = TimePeriod.^2

    df_long.TimePeriod = TimePeriod
    df_long.TimePeriod2 = TimePeriod2
    df_long.RougeUsage = RouteUsage
    df_long.RouteUsage2 = RouteUsage2
    df_long.mileage = mileage
    df_long.mileage2 = mileage2

    Y = df_long[!, :Y]

    # Fully interacted model

    θ̂_glm = glm(@formula(Y ~ mileage*mileage2*RouteUsage*RouteUsage2*TimePeriod*TimePeriod2*Branded), df_long, Binomial(), LogitLink())
    println(θ̂_glm)

    flex_logit = coef(θ̂_glm)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # a) Construct the state transition matrices

    zval,zbin,xval,xbin,xtran = create_grids()

    # b) Compute the future value terms

    # create dataframe with four variables

    odo = kron(ones(zbin),xval)
    route_usage = kron(ones(xbin), zval)
    branded = zeros(size(odo, 1))
    time = zeros(size(branded, 1))

    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])

    df_state = DataFrame()
    df_state.mileage = odo
    df_state.mileage2 = odo.^2
    df_state.RouteUsage = route_usage
    df_state.RouteUsage2 = route_usage.^2
    df_state.Branded = branded
    df_state.TimePeriod= time
    df_state.TimePeriod2 = time.^2

    # function to read in the df_fv, flexible logit estimates, and the other state variables.
    T = 20
    N = 1000

    data_parms = (β = 0.9,
                  df_state = df_state,
                  B = Branded,
                  Y = Y,
                  N = N,
                  T = T,
                  Zstate = Zstate,
                  Xstate = Xstate,
                  xtran = xtran,
                  zbin = zbin,
                  xbin = xbin,
                  xval = xval)

    function get_FV(d, theta)

        FV = zeros(zbin*xbin,2,T+1)

        for t in 1:d.T
            for b in 0:1
                d.df_state[!, :time] = repeat([t], size(d.xtran, 1))
                d.df_state[!, :branded] = repeat([b], size(d.xtran, 1))

                p_0 = [1]./([1].+exp.(predict(theta, d.df_state)))

                FV[:, b+1, t+1] = -d.β.*broadcast(log, p_0)
            end
        end

        FVT1 = zeros(d.N, d.T)

        for i in 1:d.N
            row0 = (d.Zstate[i]-1)*d.xbin+1
            for t in 1:d.T
                row1 = d.Xstate[i,t] + (d.Zstate[i]-1)*d.xbin
            
                FVT1[i, t] = (d.xtran[row1,:].-d.xtran[row0, :])'*FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]
            end
        end

        return FVT1'[:]

    end

    # c) Estimate the structural parameters

    FVT1 = get_FV(data_parms, θ̂_glm)

    df_long = @transform(df_long, :fv=FVT1)

    # use GLM to estimate the structural model
    
    theta_hat_ccp_glm = glm(@formula(Y~Odometer+Branded),
                                df_long, Binomial(), LogitLink(),
                                offset=df_long.fv)


    println(theta_hat_ccp_glm)

    return nothing
end

@time main_PS6()

#=  Takes 6.743585 seconds and the estimated coefficients are:

Coefficients:
──────────────────────────────────────────────────────────────────────────
Coef.  Std. Error       z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)   1.8422    0.0324394    56.79    <1e-99   1.77862    1.90578
Odometer     -0.243758  0.00578969  -42.10    <1e-99  -0.255106  -0.232411
Branded       0.778171  0.0390183    19.94    <1e-87   0.701697   0.854646
──────────────────────────────────────────────────────────────────────────
6.743585

=#


