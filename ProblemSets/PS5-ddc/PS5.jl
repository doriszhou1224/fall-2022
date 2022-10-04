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

function main_PS5()
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# load in the data

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame) 


# create bus id variable
df = @transform(df, bus_id = 1:size(df,1)) # 1000 by 64

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
sort!(df_long,[:bus_id,:time]) # 20000 by 6

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

theta_hat = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
println(theta_hat)

# the coefficients are: θ_0 = 1.92596, θ_1 = -0.148154, θ_2 = 1.05919

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# [ Load in the data ]

url_q3 = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
df_q3 = CSV.read(HTTP.get(url_q3).body, DataFrame) 

Y = Matrix(df_q3[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

# [ convert other data frame columns to matrices]

X = Matrix(df_q3[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])

Odo = Matrix(df_q3[:, [:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])

branded = Matrix(df_q3[:,[:Branded,]])
Zst = Matrix(df_q3[:,[:Zst,]])
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: generate state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::
zval,zbin,xval,xbin,xtran = create_grids()


# zval 101 by 1. Bins by 0.01 increments from 0.25->1,25
# zbin = 101
# xval 201 by 1, mileage bins, 0.125 increments from 0->25
# xbin = 201
# xtran 20301 by 201, transition probability Matrix

# c) Compute future value terms
T = size(X, 2)
row_xtran = size(xtran, 1)
FV = reshape(zeros(row_xtran*2*(T+1)), row_xtran, 2, T+1)
# Initialize future value array, 3D array of 0s. Dim_1 = row count of xtran, Dim_2 = 2, Dim_3 = T+1 = 21

@views @inbounds function dynamic_choice(theta, zval, zbin, xval, xbin, xtran, X, Zst)

    println("In here")
    theta_0 = theta[1]
    theta_1 = theta[2]
    theta_2 = theta[3]

    beta = 0.9
    N_bus = size(X, 1)

    # 20301 by 2 by 21 

    # 4 nested for loops

    for t in T:-1:1
        @show t
        for b in 0:1
            for z in 1:zbin
                for x in 1:xbin
                    xtran_row = x+(z-1)*xbin   # index the row of the transition matrix

                    v_1t = [theta_0 + theta_1*xval[x] + theta_2*b] .+ xtran[xtran_row,:]'.*FV[(z-1)*xbin+1:z*xbin, b+1, t+1]
                    #@show v_1t
                    v_0t = xtran[1+(z-1)*xbin,:]'.*FV[(z-1)*xbin+1:z*xbin, b+1, t+1]
                    #@show v_0t
                    FV[xtran_row,b+1,t] = beta.*sum(log.(broadcast(exp, v_0t).+broadcast(exp, v_1t)))
                end
            end
        end
    end

   #@show FV
    # d) Construct the log likelihood

    log_like = 0

    for i in 1:N_bus
        for t in 1:T
            # if bus is replaced
            #if Y[i, t] == 0
            #    tran_idx = 1+(Zst[i]-1)*xbin
            #end

            # if bus is not replaced
            #if Y[i, t] == 1
            #    tran_idx = X[i, t] + (Zst[i]-1)*xbin
            # end

            v_1t = [theta_0 + theta_1*X[i, t] + theta_2*branded[i]] .+ xtran[X[i, t] + (Zst[i]-1)*xbin,:]'.*FV[(Zst[i]-1)*xbin+1:Zst[i]*xbin, branded[i]+1, t+1]
            v_0t = xtran[1+(Zst[i]-1)*xbin,:]'.*FV[(Zst[i]-1)*xbin+1:Zst[i]*xbin, branded[i]+1, t+1]

            P_1 = broadcast(exp, v_1t.-v_0t)./([1].+broadcast(exp, v_1t.-v_0t))
            P_0 = [1].-P_1

            log_like += Y[i, t]*sum(broadcast(log, P_1)) +(1-Y[i, t])*sum(P_0)
        end
    end

    return -log_like
end

start_theta = coef(theta_hat)
theta_dynamic_ll = optimize(theta -> dynamic_choice(theta, zval, zbin, xval, xbin, xtran, X, Zst),start_theta, LBFGS(),
                                    Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
println(theta_dynamic_ll.minimizer)

return nothing
end

