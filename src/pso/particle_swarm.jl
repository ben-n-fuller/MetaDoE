module ParticleSwarm

# Export
export search_simplex, search_hypercube

# Include other project files
include("../geom/geom.jl")
include("../geom/simplex.jl")
include("../sampling/gibbs_sampler.jl")
include("../doe/designs.jl")
include("./particles.jl")

# Local modules
using .Designs
using .Simplex
using .GibbsSampler
using .Geom
using .Particles

# Packages - some of these can be taken out (maybe SpecialFunctions, Random, and DataFrames). Will need testing to do.
using Distributions, LinearAlgebra, Random, SpecialFunctions, DataFrames

function search_simplex(;
    N::Int64,                       # := Number of replicate points in design
    K::Int64,                       # := Number of design factors
    S::Int64,                       # := Number of particles
    objective,                      # := function to be minimized
    order = nothing,
    max_iter     = 1000,
    w            = 1/(2*log(2)),
    c1           = 0.5 + log(2),
    c2           = 0.5 + log(2),
    nn           = 3,
    printProg    = true,
    PSOversion   = "gbest",
    relTol       = 0,
    maxStag      = 500,
    vmaxScale    = 2,
    L            = undef,
    U            = undef)

    ## check if PSO call works
    if ~(PSOversion in ["gbest", "lbest"])
        error("PSOversion must be gbest or lbest")
    end

    no_bounds = (L == undef) & (U == undef)
    if ~no_bounds
        ## investigate the hyperpolytope
        a, b               = Geom.check_polyhedron_consistency(L, U)
        Geom.countVEF(a,b)
        centroid, vertices = Geom.compute_centroid(a,b)
        Dmat               = Simplex.vec_outer_d_mat(vertices)
        maxd               = maximum(Dmat)/(2*vmaxScale)
        use_bounds         = true
    else
        # maxd is particle step size
        # simplexMaxD is from centroid to vertex (too 4 decimals)
        #  so step size is factor 1/vmaxScale of that
        maxd       = Simplex.max_d(K)/vmaxScale
        a          = undef
        b          = undef
        centroid   = undef
        use_bounds = false
    end

    ## initialize swarm
    X, V, f, p_best, f_pbest, g_best, f_gbest, l_best, f_lbest, l_bestIndex, neighborHoods = initialize_swarm_simplex(; N = N, K = K, S = S, objective = objective, order = order, nn = nn, maxd = maxd, use_bounds = use_bounds, a = a, b = b, centroid = centroid)

    ## indicator for no imporovement in g_best
    improvement = false
    ## set stagnation counter
    stagnation = 0
    ## particle iteration set
    set = [1:1:S;]

    ## set iteration
    iter = 1
    if printProg
        print("\n", iter)
    end

    ## reltol convergence
    reltol_conv    = false
    reltol         = eps(Float64)#eps(Float64)^(0.8) #sqrt(eps(Float64))
    pbest_fit_iter = typemax(Float64)


    while iter < max_iter && stagnation < maxStag && ~reltol_conv
        iter += 1
        if printProg
            print("\n", iter)
        end

        if ~improvement
            ## generate Neighborhood communication if no improvement in g_best
            #print("======= New Neighbors =======")
            neighborHoods = gen_neighbors_simplex(S, nn)
            stagnation += 1
        else
            #neighborHoods[iter] = neighborHoods[iter - 1]
            stagnation = 0
        end

        X_t           = deepcopy(X)
        V_t           = deepcopy(V)
        f_t           = deepcopy(f)
        p_best_t      = deepcopy(p_best)
        f_pbest_t     = deepcopy(f_pbest)
        g_best_t      = deepcopy(g_best)
        f_gbest_t     = deepcopy(f_gbest)

        if (PSOversion  == "lbest")
            l_best_t        = deepcopy(l_best)
            f_lbest_t       = deepcopy(f_lbest)
            l_bestIndex_t   = deepcopy(l_bestIndex)
            neighborHoods_t = deepcopy(neighborHoods)
        elseif (PSOversion == "gbest")
            l_best_t      = deepcopy(l_best)
            f_lbest_t     = deepcopy(f_lbest)
            l_bestIndex_t = deepcopy(l_bestIndex)
        end

        ## last iterations best value
        pbest_fit_iterprior_t = deepcopy(pbest_fit_iter)
        f_gbest_prev = deepcopy(f_gbest_t)

        ## swarm knowledge update
        pset = shuffle(set)
        for i in pset
            ## update position and velocity
            X_t[:, :, i], V_t[:, :, i] = Particles.update_velocity_and_position(X_t[:, :, i], V_t[:, :, i], p_best_t[:, :, i], l_best_t[:,:,i], g_best_t, w, c1, c2, PSOversion, maxd, use_bounds, centroid, a, b)

            ## update fitness
            f_t[i] = objective(X_t[:, :, i], order = order)
        end

        ## swarm knowledge update
        for i in pset
            ## update personal and global update bests
            p_best_t[:, :, i], f_pbest_t[i], g_best_t, f_gbest_t = Particles.update_pg_bests(X_t[:, :, i], f_t[i], p_best_t[:, :, i], f_pbest_t[i], g_best_t,  f_gbest_t)
        end

        ## local neighborhood update
        if PSOversion == "lbest"
            for i in pset
                l_best_t[:, :, i], f_lbest_t[i], l_bestIndex_t[i] = Particles.update_l_best(p_best_t, f_pbest_t, l_best_t[:,:,i], f_lbest_t[i], l_bestIndex_t[i], neighborHoods_t[i])
            end
        end

        X       = deepcopy(X_t)
        V       = deepcopy(V_t)
        f       = deepcopy(f_t)
        p_best  = deepcopy(p_best_t)
        f_pbest = deepcopy(f_pbest_t)
        g_best  = deepcopy(g_best_t)
        f_gbest = deepcopy(f_gbest_t)

        if PSOversion  == "lbest"
            l_best      = deepcopy(l_best_t)
            f_lbest     = deepcopy(f_lbest_t)
            l_bestIndex = deepcopy(l_bestIndex_t)
        end

        ## iteration checks
        if f_gbest_t == f_gbest_prev
            improvement = false
        else
            improvement = true
        end

        ## reltol checker
        if ~(relTol == 0)
            pbest_fit_iter = minimum(f_t)
            rdiff = abs(pbest_fit_iterprior_t - pbest_fit_iter)
            #println("")
            #println(rdiff)
            if rdiff == 0
                #println("(0)rdiff = ", rdiff)
                reltol_conv = false
            else
                reltol_conv = rdiff <= reltol
                #println("rdiff = ", rdiff)
            end
        end

    end

    return iter, S, f_gbest, g_best

end

function initialize_swarm_simplex(; N, K, S, objective, order = nothing, nn, maxd, use_bounds, a, b, centroid)
    ## swarm swarm_initialization
    ## dimensions are
    X0 = Array{Float64}(undef, (N, K, S))
    V0 = deepcopy(X0)
    # populate the particle array and velocity array
    #nInt = S - 10
    if ~use_bounds
        for i in 1:S
            X0[:, :, i] = Designs.generate_mixture_design(N, K)
            vt          = Designs.generate_mixture_design(N, K)
            nvt         = Simplex.norm(vt)
            if nvt > maxd
                vt = Simplex.scale(vt, maxd)
            end
            V0[:, :, i] = vt
        end
    else
        for i in 1:S
            X0[:, :, i] = GibbsSampler.sample_truncated_dirichlet(N, a, b, undef, centroid)
            vt          = Designs.generate_mixture_design(N, K)
            nvt         = Simplex.norm(vt)
            if nvt > maxd
                vt = Simplex.scale(vt, maxd)
            end
            V0[:, :, i] = vt
        end
    end

    ## evaluate the objective
    f0 = Vector(undef, S)
    if isnothing(order)
        for i in 1:S
            f0[i] = objective(X0[:, :, i])
        end
    else
        for i in 1:S
            f0[i] = objective(X0[:, :, i], order = order)
        end
    end
    ## personal bests
    p_best0      = deepcopy(X0)
    f_pbest0     = fill(typemax(Float64), S)
    ## global bests
    g_best0      = deepcopy(X0[:, :, 1])
    f_gbest0     = typemax(Float64)
    ## local bests
    neighbors    = gen_neighbors_simplex(S, nn)
    l_best0      = deepcopy(X0)
    f_lbest0     = fill(typemax(Float64), S)
    l_bestIndex0 = collect(1:1:S)
    ## local neighborhood update
    for i in collect(1:1:S)
        l_best0[:, :, i], f_lbest0[i], l_bestIndex0[i] = Particles.update_l_best(p_best0, f_pbest0, l_best0[:,:,i], f_lbest0[i], l_bestIndex0[i], neighbors[i])
    end

    return X0, V0, f0, p_best0, f_pbest0, g_best0, f_gbest0, l_best0, f_lbest0, l_bestIndex0, neighbors
end

## generate neighborhood communication links
function gen_neighbors_simplex(S, nn = 3)
    # NOTE: this function generates a list of length S,
    #       the ith element is the list of neighbors of particle i
    p_avg = 1 - (1-1/S)^nn
    d     = Uniform(0,1)
    tmp   = reshape(rand(d, S*S) .< p_avg, (S, S))
    # connect particles to themselves
    tmp[diagind(tmp)] .= 1

    Neighbors = Array{Any}(undef, S)
    for i in 1:S
        indtemp  = findall(tmp[:, i])
        l        = length(indtemp)
        Neighbors[i] = indtemp
    end
    return Neighbors
end


###################### Begin Hypercube Code ################################

function search_hypercube(;
    N::Int64,                      # := Number of replicate points in design
    K::Int64,                      # := Number of design factors
    S::Int64,                      # := Number of particles
    objective,              # := fuction to be minimized
    l_vec,                  # := k x 1 vector of lower bounds on design factors
    u_vec,                  # := k x 1 vector of upper bounds on design factors
    max_iter = 5000,        # := max number of PSO update iterations (alg will terminate)
    w::Float64 = 1/(2*log(2)),       # := HAL 2012 scaling factor on inertia component
    c1::Float64 = 0.5 + log(2),      # := HAL 2012 scaling factor on cognitive component
    c2::Float64 = 0.5 + log(2),      # := HAL 2012 scaling factor on social component
    PSOversion = "Basic",   # := HAL 2012 versions available, \in {Basic, 2007}
    nn = 3,                 # := expected number of neighbor links in communication topology
    maxStag = 250,          # := max number of iter stagnations of soln b/f alg terminates
    init_method = "random", # := starting positions should be random, or filled via LHS,
                            #    permissible values are "random" or "matrixFill"
    ngen = 1,               # := number of generations to make RLHC matrix space filling,
                            #    for large k, N, S need this small or it takes a lot of time
    printProg = true,       # := logical, should PSO iteration print while running algorithm?
    output    = "small",
    relTol    = 0,
    order = NaN)            # := in [NaN (vector inputs),
        #                           0 (matrix inputs first order no interactions),
        #                           1 (matrix inputs first order with interactions),
        #                           2 (matrix inputs second order)]

    ##  -------------------------------------------------------------------------------------------
    #       BEGIN PSO FOR DoE!! XD
    ##  -------------------------------------------------------------------------------------------

    ## input checking
    if !isnan(NaN)
        ~(order in [0 1 2]) && error("(swarmSearch_Hypercube) only model order 0, 1, or 2 is currently supported for design searches")
        ~(K in [1 2 3 4 5]) && error("(swarmSearch_Hypercube) only k in [1, 2, 3, 4, 5] is currently supported")
    end
    # other constants
    #U = Uniform(0, 1)
    v_max = u_vec - l_vec
    ## allocate memory to record swarm info
    allData       = Array{Any}(undef, 13)

    ## initialize swarm
    X, V, f, p_best, f_pbest, g_best, f_gbest, l_best, f_lbest, l_bestIndex, neighborHoods = initialize_swarm_hypercube(; N = N, k = K, S = S, objective = objective, l_vec = l_vec, u_vec = u_vec, PSOversion = PSOversion, nn = nn, init_method = init_method, ngen = ngen, order = order)

    ## set confinement method
    if PSOversion == "Basic"
        confine_particle = confine_particle_B2007
        update_Vel = update_Vel_Basic
    elseif PSOversion == "2007"
        confine_particle = confine_particle_B2007
        update_Vel = update_Vel_2007
    else
        stop("PSOversion must be in [Basic, 2007]")
    end

    ## indicator for no imporovement in g_best
    improvement = false
    ## set stagnation counter
    stagnation  = 0
    ## particle iteration set
    set = [1:1:S;]
    pset = deepcopy(set)
    ## set iteration
    iter = 1
    if printProg
        print("\n", iter)
    end
    ## reltol convergence
    reltol_conv    = false
    reltol         = eps(Float64)^(0.6) #sqrt(eps(Float64))
    pbest_fit_iter = typemax(Float64)

    while iter < max_iter && stagnation < maxStag && ~reltol_conv
        iter += 1
        if printProg
            print("\n", iter)
        end

        pbest_fit_iterprior_t = deepcopy(pbest_fit_iter)
        f_gbest_prev = deepcopy(f_gbest)
        if ~improvement
            ## generate Neighborhood communication if no improvement in g_best
            #print("======= New Neighbors =======")
            if ~(PSOversion  == "Basic")
                neighborHoods = gen_neigbors_hypercube(S, nn)
            end
            stagnation += 1
        else
            if ~(PSOversion  == "Basic")
                neighborHoods = neighborHoods
            end
            stagnation = 0
        end


        ## THIS IS SYNCHRONOUS UPDATE, ---------------------------------------------------------
        #   demonstrated to be superior to asynchronous update by
        #    Rada-Vilela, Zhang, and Seah (2011)
        ## velocity, position and fitness update
        for i in pset
            ## update position and velocity
            V[:, :, i] = update_Vel(X[:, :, i], V[:, :, i], p_best[:, :, i], l_best[:,:,i], g_best, l_vec, u_vec,  w, c1, c2; N = N, k = K)

        end

        # for speed limit velocities on total array
        any_outv = check_velocity(; V = V, l_vec = l_vec, u_vec = u_vec, k = K)
        if any_outv
            #println("")
            #println("VELOCITIES LIMITED")
            for i in pset
                V[:, :, i] = velLimit(V[:, :, i], l_vec, u_vec; k = K)
            end
        end

        # move the particles
        X = X + V

        # try to reduce calls to confine by checking before if needed
        any_outx = check_confinement(; X = X, l_vec = l_vec, u_vec = u_vec, k = K)
        if any_outx
            #println("")
            #println("BOUNDARIES VIOLATED")
            for i in pset
                ## confinement
                X[:, :, i], V[:, :, i] = confine_particle(; X = X[:, :, i], V = V[:, :, i], l_vec = l_vec, u_vec = u_vec, k = K)
            end
        end

        ## update fitness and personal/global bests
        if isnan(order)
        #    # NOTE: this condition is for functions with vector inputs
            for i in pset
                f[i] = objective(X[:, :, i])
                p_best[:, :, i], f_pbest[i], g_best, f_gbest = update_pg_bests(X[:, :, i], f[i], p_best[:, :, i], f_pbest[i], g_best,  f_gbest)
            #end
            end
        else
        #    # NOTE: this condition is for functions with matrix inputs
            for i in pset
                f[i] = objective(X[:, :, i]; N = N, K = K, order = order)
                p_best[:, :, i], f_pbest[i], g_best, f_gbest = update_pg_bests(X[:, :, i], f[i], p_best[:, :, i], f_pbest[i], g_best,  f_gbest)
            #end
            end
        end


        ## local neighborhood update - this uses the vector f_pbest
        #   and so needs to by run (sync) after all particles have moved
        # NOTE: doesnt need to be run if using global communication topology
        if ~(PSOversion  == "Basic")
            for i in pset
                l_best[:, :, i], f_lbest[i], l_bestIndex[i] = update_l_best(p_best, f_pbest, l_best[:,:,i], f_lbest[i], l_bestIndex[i], neighborHoods[i])
            end
        end


        #println("")
        #println(f_gbest)
        ## check for improvement in solution
        if f_gbest == f_gbest_prev
            improvement = false
        else
            improvement = true
        end
        ## reltol checker
        if ~(relTol == 0)
            pbest_fit_iter = minimum(f)
            rdiff = abs(pbest_fit_iterprior_t - pbest_fit_iter)
            #println("")
            #println(rdiff)
            if rdiff == 0
                #println("(0)rdiff = ", rdiff)
                reltol_conv = false
            else
                reltol_conv = rdiff <= reltol
                #println("rdiff = ", rdiff)
            end
        end

    end ## end PSO while loop iteration

    info = "This array contains: \n      [2]  X \n      [3]  V \n      [4]  f \n      [5]  p_best \n      [6]  f_pbest \n      [7]  g_best \n      [8]  f_gbest \n      [9]  l_best \n      [10] f_lbest \n      [11] l_bestIndex \n      [12] neighborhoods \n      [13] n(iterations)"
    allData[1]  = info
    allData[2]  = X
    allData[3]  = V
    allData[4]  = f
    allData[5]  = p_best
    allData[6]  = f_pbest
    allData[7]  = g_best
    allData[8]  = f_gbest
    if ~(PSOversion  == "Basic")
        allData[9]  = l_best
        allData[10] = f_lbest
        allData[11] = l_bestIndex
        allData[12] = neighborHoods
    else
        allData[9]  = "neighborhoods not used in Gbest topology"
        allData[10] = "neighborhoods not used in Gbest topology"
        allData[11] = "neighborhoods not used in Gbest topology"
        allData[12] = "neighborhoods not used in Gbest topology"
    end
    allData[13] = iter

    if output == "small"
        return iter, S, f_gbest, g_best
    else
        return allData
    end
end

function initialize_swarm_hypercube(; N, k, S, objective, l_vec, u_vec, PSOversion = "2007", nn, init_method = "random", ngen = 1, order = NaN)
    ## NOTE: swarm initialization
    #
    #       N           := number of experimental replicates
    #       k           := dimension of the search space
    #       S           := number of particles
    #       l_vec       := k x 1 vector of lower bounds of search space
    #       u_vec       := k x 1 vector of upper bounds of search space
    #       PSOversion  := permissible text strings are (Basic, 2007)
    #       nn          := expected number of neighbors in communication topology
    #       init_method := random
    #       ngen        := control parameter to optimize RLHCS, expensive for
    #                      high dimension
    #       order       := in [NaN (vector inputs),
    #                           0 (matrix inputs first order no interactions),
    #                           1 (matrix inputs first order with interactions),
    #                           2 (matrix inputs second order)]

    # input checks --------
    ~(PSOversion in ["Basic", "2007"]) && error("(initialize_swarm_hypercube) PSOversion must be in ['Basic', '2007']")
    ~(init_method in ["random", "matrixFill"]) && error("(initialize_swarm_hypercube) init_method must be in ['random', 'matrixFill']")
    ~(isnan(order) | (order in [0 1 2])) && error("(initialize_swarm_hypercube) order must be in [NaN, 0, 1, 2]")

    # set velocity initialization
    if PSOversion in(["Basic" "2007"])
        initialVelocity = initialVelocity_2007
    end

    # allocate space for positions and velocities
    X0 = Array{Float64}(undef, (N, k, S))
    V0 = Array{Float64}(undef, (N, k, S))

    # populate the particle array
    if init_method == "random"
        for i in 1:S
            # genRandDesign_fac generates a single random design matrix, so need a loop
            X0[:, :, i] = genRandDesign_fac(N = N, k = k, l_vec = l_vec, u_vec = u_vec )
        end
    elseif init_method == "matrixFill"
        for i in 1:S
            Xtt , _       = LHCoptim(N, k, ngen)
            scale_range   = Vector{Any}(undef, k)
            for i in 1:k
                scale_range[i] = [l_vec[i], u_vec[i]]
            end
            scale_range = Tuple.(scale_range)
            # genRandDesign_fac generates a single random design matrix, so need a loop
            X0[:, :, i] = scaleLHC(Xtt, scale_range)
        end

    end

    ### round to 4 spots right off
    #X0 = round.(X0, digits = 4)


    # populate velocity array
    for i in 1:S
        V0[:, :, i] = velLimit(initialVelocity(X0[:, :, i], l_vec = l_vec, u_vec = u_vec), l_vec, u_vec, k = k)
    end

    ## evaluate the fitness of each particle
    f0 = Vector{Float64}(undef, S)
    if isnan(order)
        # NOTE: this condition is for functions with vector inputs
        for i in 1:S
            f0[i] = objective(X0[:, :, i])
        end
    else
        # NOTE: this condition is for functions with matrix inputs
        for i in 1:S
            f0[i] = objective(X0[:, :, i]; N = N, K = k, order = order)
        end
    end

    ## personal bests
    p_best0      = deepcopy(X0)
    f_pbest0     = deepcopy(f0)
    ## global bests
    g_best0      = X0[:, :, argmin(f_pbest0)]
    f_gbest0     = f_pbest0[argmin(f_pbest0)]
    ## local bests
    neighbors    = gen_neigbors_hypercube(S, nn)
    l_best0      = deepcopy(X0)
    f_lbest0     = fill(typemax(Float64), S)
    l_bestIndex0 = collect(1:1:S)
    ## local neighborhood update
    for i in collect(1:1:S)
        l_best0[:, :, i], f_lbest0[i], l_bestIndex0[i] = update_l_best(p_best0, f_pbest0, l_best0[:,:,i], f_lbest0[i], l_bestIndex0[i], neighbors[i])
    end

    return X0, V0, f0, p_best0, f_pbest0, g_best0, f_gbest0, l_best0, f_lbest0, l_bestIndex0, neighbors
end

function gen_neigbors_hypercube(S, nn = 3)
    # NOTE: this function generates a list of length S,
    #       the ith element is the list of neighbors of particle i
    #       the particle itself is always included in the list
    #
    #  S  := number of particles
    #  nn := expected number of neighbors
    p_avg = 1 - ( 1 - 1 / S )^nn
    #d     = Uniform(0,1)
    #tmp   = reshape(rand(d, S*S) .< p_avg, (S, S))
    #Rm   = rand(d, S, S)
    Rm   = rand(S,S)
    tmp  = Rm .< p_avg
    # connect particles to themselves
    tmp[diagind(tmp)] .= 1
    # allocate space to hold neighborhood
    Neighbors = Array{Any}(undef, S)
    for i in 1:S
        indtemp      = findall(tmp[:, i])
        #indtemp      = findall(x -> x == 1, tmp[:, i])
        Neighbors[i] = indtemp
    end
    return Neighbors
end

end # module


