#= Assignment 1 

Xin Yue Zhou
113357760

=#

# Load packages

using CSV
using DataFrames
using Distributions
using FreqTables
using JLD2
using LinearAlgebra
using Random
using Statistics
using Primes

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# a)

# Set the seed to 1234. Create 4 matrices-- A, B, C, D
function q1()
    Random.seed!(1234)

#= First matrix (A) is 10 by 7 with random numbers distributed on the uniform distribution U[-5, 10]
   Second matrix (B) is 10 by 7 drawn from N(-2, 15)
   Third matrix (C) is 5 by 7
   Fourth matrix (D) is 10 by 7. 
=#

    A = rand(Uniform(-5,10), 10, 7)
    B = rand(Normal(-2, 15), 10,7)
    # sd*randn(n, m).+mean

    sub_A = A[1:5, 1:5]
    sub_B = B[1:5, 6:7]
    C = [sub_A sub_B]

    D .= A
    D[D .> 0] .= 0

# b) Print the number of elements in A. Should be 70 elements.

    A_length = length(A)
    println(A_length)

# c) List number of unique elements in D

    D_unique = unique(D)
    println(length(D_unique))

 # d) Create matrix E, vectorized from B, with the reshape() function.

    E = reshape(B, 1, length(B))
 
 # This is easier way, to just use the inbuilt function vec:
 # vec(B)

 # e) Create array F. Has A in the first column of the third dimension
 # and B in the second column of the third dimension.

    F = cat(A, B, dims=3)

 # f) Twist F with permutedims() so it is 2 by 7 by 10 and save as f

    F = permutedims(F, (3,1,2))

 # g) Create matrix G = B ⊗ C

    G = kron(B, C)

 # Try C ⊗ f
 # kron(C, F)
 # Gives MethodError for types matrix and array in kron() function

 # h) Save the matrices A, B, C, D, E, F, and G as .jld file called matrixpratice.

    @save "ProblemSets/PS1-julia-intro/matrixpratice.jld2" A B C D E F G

 # i) Save matrices A,B, C, D as jld file called firstmatrix.

    @save "ProblemSets/PS1-julia-intro/firstmatrix.jld2" A B C D

 # j) Export C into Cmatrix.csv.

    C_df = DataFrame(C, :auto)
    CSV.write("ProblemSets/PS1-julia-intro/Cmatrix.csv", C_df)

 # k) Export D as tab-delimited file Dmatrix.dat
 
    D_df = DataFrame(D, :auto)
    CSV.write("ProblemSets/PS1-julia-intro/Dmatrix.dat", D_df) # delim = "\t" 

    return A, B, C, D
end

 # l) Wrap function around all code above. 

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;:
# Question 2

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# a) Compute the element-by-element product of A and B. Call the new matrix AB and call AB2
# the matrix that does this without a loop or comprehension.

function q2(A, B, C)
    row = size(A, 1)
    col = size(A, 2)
    AB = zeros(row, col)
    for i=1:row, j=1:col
        #println(typeof(B[i][j]))
        AB[i,j] = A[i,j]*B[i,j]
    end
    # with no loop
    AB2 = A.*B

    # b) Write loop to create column vector Cprime for prime numbers ∈[-5, 5] 
    # and create vector Cprime2 without a loop.

    prime_range = collect(-5:5)
    Cprime = []
    not_prime = []
    for i in 2:maximum(prime_range)-1
        for r in prime_range
            if r == i || r == -i
                continue
            end
            if  r%i == 0 || r == 1 || r == -1
                push!(not_prime, r)
            end
        end
    end

    Cprime = prime_range[prime_range .∉ Ref(not_prime)]

    # without a loop

    temp_prime = filter(Primes.isprime, prime_range)
    Cprime2 = [temp_prime; -1*temp_prime]

    # c) Create 3d array called X.

    N = 15169
    K = 6
    T = 5

    
    X_N1T = ones(N, T) # matrix of ones
    
    col2_list = []
    for t in 1:T
        p = Binomial(1, 0.75*(6-t)/5)
        X_t = rand(p, N)

        push!(col2_list, X_t)
    end

    X_N2T = reduce(hcat, col2_list) # matrix with 1s of a certain probability dependent on T
    
    col3_list = []
    for t in 1:T
        X_t = rand(Normal(15+t-1, 5*(t-1)), N)
        push!(col3_list, X_t)
    end

    X_N3T = reduce(hcat, col3_list) # matrix drawn from normal distribution with mean and s.d. dependent on T

    col4_list = []
    for t in 1:T
        X_t = rand(Normal(π*(6-t)/3, 1/ℯ), N)
        push!(col4_list, X_t)
    end

    X_N4T = reduce(hcat, col4_list) # matrix drawn from another normal distribution

    col5_list = []
    for t in 1:T
        X_t = rand(Binomial(20, 0.6), N)
        push!(col5_list, X_t)
    end

    X_N5T = reduce(hcat, col5_list) # matrix drawn from discrete Normal

    col6_list = []
    for t in 1:T
        X_t = rand(Binomial(20, 0.5), N)
        push!(col6_list, X_t)
    end

    X_N6T = reduce(hcat, col6_list) # matrix drawn from a different discrete Normal

    X = permutedims(cat(X_N1T, X_N2T, X_N3T, X_N4T, X_N5T, X_N6T, dims=3), (1, 3, 2))

    # d) Create matrix β, K × T

    β_1T = [1+0.25*t for t in 0:T-1]

    β_2T = [log(t) for t in 1:T]

    β_3T = [-sqrt(t) for t in 1:T]

    β_4T = [ℯ^t-ℯ^(t+1) for t in 1:T]

    β_5T = [t for t in 1:T]

    β_6T = [t/3 for t in 1:T]

    β = permutedims(reduce(hcat, (β_1T, β_2T, β_3T, β_4T, β_5T, β_6T)), (2,1))

    # e) Create a matrix Y which is N × T

    #ϵ = rand(Normal(0, 0.36), N, T)

    Y = reduce(hcat, [X[:,:,t]*β[:,t]+rand(Normal(0, 0.36), N, 1) for t in 1:T])

    # f) wrap question 2 in a function q2()

    return nothing
end

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
function  q3()

    # a) Read in the nlsw88.csv as a dataframe and then save as .jld file.

    nlsw88 = CSV.read("ProblemSets/PS1-julia-intro/nlsw88.csv", DataFrame)
    # nlsw88 = dropmissing(nlsw88)
    @save "ProblemSets/PS1-julia-intro/nlsw88.jld2" nlsw88

    # b) Look at percentage of the sample that has never been married. Also look are percentage of college graduates. 

    # show(nlsw88, allcols=true)

    married_vec = nlsw88[!, "married"]
    never_married = length(married_vec[married_vec .==0])/length(married_vec)
    
    # There has been 34.65% in the data who has never been married. 

    coll_grad = nlsw88[!, "collgrad"]
    coll_grad_prop = length(coll_grad[coll_grad .== 1])/length(coll_grad)

    # In the data, 24.78% are college graduates.

    # c) freqtable() to report what percentage is in each race.

    freqtable(nlsw88, "race")

    # d) Use describe() function

    summarystats = describe(nlsw88, :all)

    # There are 2 missing from the grade column.

    # e) The joint distribution of industry and occupation using cross-tabulation.

    freqtable(nlsw88, "industry", "occupation")

    # f) Tabulate the mean wage over industry and occupation categories. 

    nlsw88_sub = nlsw88[:, ["industry", "occupation", "wage"]]
    grouped = groupby(nlsw88_sub, [:industry, :occupation])
    combine(grouped, :wage => mean)
    
    return nothing
end

# f) wrap function q3()

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function q4()

    # a) Load firstmatrix.jld

    @load "ProblemSets/PS1-julia-intro/firstmatrix.jld2" A B C D

    # b) Write function matrixops()

    function matrixops(A, B)

        # This function takes as input two matrices, A and B, and returns 3 different computations 
        # done with A and B. First, it returns the element by element product of A and B. Second, it
        # returns the matrix product of A and B, which is 7 × 7. Third, it returns the sum of the sum
        # of A and B. 

        if size(A) != size(B)
            println("inputs must have the same size")
            return nothing, nothing, nothing 
        end

        A_element_B = A .* B
        A_transp_B = transpose(A)*B
        sum_A_B = sum(A + B)

        return A_element_B, A_transp_B, sum_A_B
        
    end

    # d) Evaulate matrixops() with A and B from firstmatrix.jld

    AB_elprod, AB_prod, AB_sum = matrixops(A, B)

    # e) Add if statement in matrixops

    # f) Evalute matrixops with different size matrices C and D.

    CD_elprod, CD_prod, CD_sum = matrixops(C, D)

    # The function prints the error message and I return three nothing results.

    # g) Evaluate matrixops with ttl_exp and wage from nlsw88.jld

    @load "ProblemSets/PS1-julia-intro/nlsw88.jld2" nlsw88

    ttl_exp_vec = convert(Array, nlsw88.ttl_exp)
    wage_vec = convert(Array, nlsw88.wage)

    tw_elprod, tw_prod, tw_sum = matrixops(ttl_exp_vec, wage_vec)

    # Both vectors are of the same size, so the function returns three results that are not nothing.

    # h) Wrap in function q4()
end

A, B, C, D = q1()
q2(A, B, C)
q3()
q4()






