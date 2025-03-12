module Particles

export update_velocity_and_position, update_l_best, update_pg_bests, confine_particle_B2007, update_Vel_Basic, update_Vel_2007, check_velocity, velLimit, check_confinement, initialVelocity_2007, genRandDesign_fac

using Distributions

function update_velocity_and_position(X, V, p_best, l_best, g_best, w, c1, c2, PSOversion, maxd, use_bounds, centroid, a, b)
    u    = Uniform(0, 1)
    Vnew = deepcopy(V)
    Xnew = deepcopy(X)
    # update the rows independently
    msize = size(X)
    nrow  = msize[1]
    K     = msize[2]
    alpha = fill(1, K)
    d     = Dirichlet(alpha)


    # pick the correct comparitor for the chosen communication topology
    if PSOversion == "gbest"
       groupbest = g_best
    elseif PSOversion == "lbest"
       groupbest = l_best
    end
    rset = [1:1:nrow;]
    pset = shuffle(rset)
    for j in pset
        ## update velocity
        # inertia
        inertia   = Simplex.multiply(V[j,:], w)
        # cognitive
        ct1       = Simplex.multiply(X[j,:], -1.0)
        ct2       = Simplex.add(p_best[j,:], ct1)
        ctscaler  = c1*rand(u)
        cognitive = Simplex.multiply(ct2, ctscaler)
        # social
        st1       = Simplex.multiply(X[j,:], -1.0)
        st2       = Simplex.add(groupbest[j,:], st1)
        sscaler   = c2*rand(u)
        social    = Simplex.multiply(st2, sscaler)
        # velocity update
        vt        = Simplex.add(inertia, Simplex.add(cognitive, social))
        nvt       = Simplex.norm(vt)
        if nvt > maxd
            vt = Simplex.scale(vt, maxd)
        end
        Vnew[j,:] = vt
        if ~use_bounds
            Xnew[j,:] = Simplex.add(Xnew[j,:], Vnew[j,:])
        else
            Xnt       = Simplex.add(Xnew[j,:], Vnew[j,:])
            Xnew[j,:] = Geom.hyper_polyhedron_confine(transpose(Xnt), centroid, a, b)
        end
    end
    return Xnew, Vnew
end

## update local update_pg_bests - done for particle i
function update_l_best(p_best, f_pbest, l_best, f_lbest, lbest_index, neighbors)
    # This one is a bit tricky, so here are the arugment definitions
    # NOTE:
    #       1. if input is vector or array, it is the full array (all particles)
    #          at iteration time t
    #
    #  p_best  := array of p_best locations across all particles at time t
    #  f_pbest := vector of fitness of all p_best's at time t
    #  l_best  := the local neighborhood best location for particle i (single location)
    #  f_lbest := fitness at l_best (single value)
    #  lbest_index := single value indicating the index of this particles
    #                 best neighbor in its neighbor set
    #  neighbors := particle i's vector of neighbor indices

    #  compute which neighbor currently has the best p_best location
    best_neighb_index   = argmin(f_pbest[neighbors])

    if f_pbest[neighbors][best_neighb_index] < f_lbest
        # which neighbor of particle i has the best position
        lbest_index  = neighbors[best_neighb_index]
        # what is the position of the best nerighbor
        l_best       = p_best[:, :, lbest_index]
        # what is the fitness at the best neighborhood position
        f_lbest      = f_pbest[lbest_index]
    end
    return l_best, f_lbest, lbest_index
end

function update_pg_bests(x, f,  p_best, f_pbest, g_best, f_gbest)
    # knowledge check
    if f < f_pbest
        f_pbest = f
        p_best = x
        if f_pbest < f_gbest
            f_gbest = f_pbest
            g_best  = p_best
        end
    end
    return p_best, f_pbest, g_best, f_gbest
end

##################### Hypercube specific ############################

function confine_particle_B2007(; X, V, l_vec, u_vec, k)
    # absorbing wall
    #println("ranAbswall")
    for j in 1:k
        #out_bottom = findall(x -> x <= l_vec[j], X[:, j])
        out_bottom = straysBottom(l_vec[j], X[:, j])
        X[out_bottom, j] .= l_vec[j] # + eps()
        V[out_bottom, j] .= 0

        #out_top = findall(X[:, j] .>= u_vec[j])
        #out_top = findall(x -> x >= u_vec[j], X[:, j])
        out_top = straysTop(u_vec[j], X[:, j])
        X[out_top, j] .= u_vec[j] # - eps()
        V[out_top, j] .= 0
    end
    return X, V
end


## update velocity and position
function update_Vel_Basic(X, V, p_best, l_best, g_best, l_vec, u_vec, w, c1, c2; N, k)
    r1        = rand(N, k)
    r2        = rand(N, k)
    inertia   = @. w * V
    cognitive = @. c1 * r1 * (p_best - X)
    social    = @. c2 * r2 * (g_best - X)
    Vnew      = @. inertia + cognitive + social
    return Vnew
end

function update_Vel_2007(X, V, p_best, l_best, g_best, l_vec, u_vec, w, c1, c2; N, k)
    r1        = rand(N, k)
    r2        = rand(N, k)
    inertia   = @. w * V
    cognitive = @. c1 * r1 * (p_best - X)
    social    = @. c2 * r2 * (l_best - X)
    if l_best == p_best
        Vnew  = @. inertia + cognitive
    else
        Vnew  = @. inertia + cognitive + social
    end
    return Vnew
end



function check_velocity(; V, l_vec, u_vec, k)

    check1 = Vector{Bool}(undef, k)
    vmax = u_vec - l_vec
    for j in 1:k
        check1[j] = any(abs.(V[:,j,:]) .> vmax[j])
    end
    any_out = any(check1)
    return any_out
end

function velLimit(V, l_vec, u_vec; k)
    # V is a velocity matrix for a single particle
    vmax = u_vec - l_vec
    for j in 1:k
        vj           = vmax[j]
        #badVs        = findall(abs.(V[:, j]) .> vj)
        badVs        = findall(x -> abs(x) > vj, V[:, j])
        if length(badVs) > 0
            V[badVs, j] .= sign.(V[badVs, j]) .* vj
        end
    end
    return V
end

function check_confinement(; X, l_vec, u_vec, k)

    check1 = Vector{Bool}(undef, k)
    for j in 1:k
        check1[j] = any(X[:,j,:] .<= l_vec[j]) | any(X[:,j,:] .>= u_vec[j])
    end
    any_out = any(check1)
    return any_out
end


function initialVelocity_2007(X; l_vec, u_vec)
    # X:= design matrix
    # l_vec := vec of lower bounds for search space
    # u_vec := vec of upper bounds for search space
    msize = size(X)
    N     = msize[1]
    k     = msize[2]
    nlb   = length(l_vec)
    nub   = length(u_vec)
    (k != nlb) | (k != nub) && error("(initialVelocity_2007) dim of space (k) must equal length of bound vectors")
    U     = Matrix{Float64}(undef, N, k)
    for j in 1:k
        l = l_vec[j]
        u = u_vec[j]
        d = Uniform(l, u)
        U[:,j] = rand(d, N)
    end
    result =  0.5 .* (U - X)
    return result
end

function genRandDesign_fac(; N, k, l_vec, u_vec )
    #
    #  NOTE: this function generates a random design matrix of
    #        dimension N x k
    #
    #        N := number of design points
    #        k := dimension of space
    #        l_vec := vec of lower bounds for rUnif
    #        u_vec := vec of upper bounds for rUnif
    nlb = length(l_vec)
    nub = length(u_vec)
    (k != nlb) | (k != nub) && error("(genRandDesign_fac) dim of space (k) must equal length of bound vectors")
    X = Matrix{Float64}(undef, N, k)
    for j in 1:k
        l = l_vec[j]
        u = u_vec[j]
        d = Uniform(l, u)
        X[:,j] = rand(d, N)
    end
    return X
end

function straysBottom(l, x)
    result = (1:length(x))[x .<= l]
    return result
end

function straysTop(u, x)
    result = (1:length(x))[x .>= u]
    return result
end

end