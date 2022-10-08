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


function wrapper()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: read in data
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body,DataFrame)

    # create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

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
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])
    
    
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: estimate a static version of the model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    θ̂_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    println(θ̂_glm)
    
    
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3a: read in data for dynamic model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body,DataFrame)
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df[:,:RouteUsage])
    B = Vector(df[:,:Branded])
    N = size(Y,1)
    T = size(Y,2)
    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3b: generate state transition matrices
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    zval,zbin,xval,xbin,xtran = create_grids()
    
    # create a named tuple that stores everything so we can cut down on the number of function inputs
    data_parms = (β = 0.9,
                  Y = Y,
                  B = B,
                  N = N,
                  T = T,
                  X = X,
                  Z = Z,
                  Zstate = Zstate,
                  Xstate = Xstate,
                  xtran = xtran,
                  zbin = zbin,
                  xbin = xbin,
                  xval = xval)
    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3c: write likelihood function
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @views @inbounds function likebus(θ,d)
    
        # First loop: solve the backward recursion problem given the values of θ
        # This will give the future value for *all* possible states that we might visit
        # This is why the FV array does not have an individual dimension
        
        # initialize FV in period T+1 = 0 since this is a finite horizon problem
        FV=zeros(d.zbin*d.xbin,2,T+1)
        
        for t=d.T:-1:1
            for b=0:1
                for z=1:d.zbin
                    for x=1:d.xbin
                        # inputs to FV
                        row = x + (z-1)*d.xbin                                                                               # which row of xtran should we look at? depends on milage bin x and route usage z
                        v1  = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[           row,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin,b+1,t+1] # mileage bin is x, route usage z is permanent
                        v0  =                                  d.xtran[1+(z-1)*d.xbin,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin,b+1,t+1] # the engine got replaced => mileage is 0, so first bin
                        
                        # FV is discounted log sum of the exponentiated conditional value functions
                        FV[row,b+1,t] = d.β*log(exp(v1) + exp(v0))
                    end
                end
            end
        end 

        # Second loop: form the likelihood given the future values implied by the previous θ guess
        # Here, we will take the state-specifc FV's calculated in the first loop
        # But we will only use in the likelihood those state value that are actually visited
        like = 0
        for i=1:d.N
            row0 = (d.Zstate[i]-1)*d.xbin+1 # this is the same argument as the index of xtran in v0 above, but we use the actual Z
            for t=1:d.T
                row1  = d.Xstate[i,t] + (d.Zstate[i]-1)*d.xbin                                                                      # this is the same as row in the first loop, except we use the actual X and Z
                v1    = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i] + (d.xtran[row1,:].-d.xtran[row0,:])⋅FV[row0:row0+d.xbin-1,d.B[i]+1,t+1] # using the same formula as v1 above, except we use observed values of X and B, and we difference the transitions
                dem   = 1 + exp(v1)
                like -= ( (d.Y[i,t]==1)*v1 ) - log(dem) # negative of the binary logit likelihood
            end
        end
        return like
    end
    
    θ_start = rand(3)
    θ_true  = [2; -.15; 1]
    # how long to evaluate likelihood function once?
    println("Timing (twice) evaluation of the likelihood function")
    @time likebus(θ_start,data_parms)
    @time likebus(θ_start,data_parms)
    # estimate likelihood function
    θ̂_optim = optimize(a -> likebus(a,data_parms), θ_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    θ̂_ddc = θ̂_optim.minimizer
    println(θ̂_ddc)
    
    return nothing
end
wrapper()