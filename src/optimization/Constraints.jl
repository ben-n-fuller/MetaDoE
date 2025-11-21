module Constraints

using LinearAlgebra
using Polyhedra
using HiGHS
using LinearAlgebra
using NPZ
using ..Experiments
using ..TensorOps

function hypercube(n::Int, k::Real)
    I_n = Matrix{Float64}(I, n, n)
    A = vcat(I_n, -I_n)
    b = vcat(fill(Float64(k), n), fill(Float64(k), n))
    return A, b
end

function simplex(K)
    # Non-negativity
    A_nonneg = -I(K)

    # Sum-to-one
    A_sum_upper = ones(1, K)
    A_sum_lower = -ones(1, K)

    # Combine constraints
    A = vcat(A_nonneg, A_sum_upper, A_sum_lower)
    b = vcat(zeros(K), [1.0], [-1.0])

    return A, b
end

function validate(exp::Experiments.Experiment)
    lib = DefaultLibrary{Float64}(HiGHS.Optimizer)
    A, b = exp.constraints.A, exp.constraints.b
    p = polyhedron(hrep(A, b), lib)
    try 
        center = center_of_mass(p)
        return center
    catch
        error("The constraints are infeasible. Please check the constraints.")
    end
end

function get_vertices(experiment::Experiments.Experiment)
    lib = DefaultLibrary{Float64}(HiGHS.Optimizer)
    p = polyhedron(hrep(experiment.constraints.A, experiment.constraints.b), lib)
    verts = vrep(p)
    verts = collect(points(verts))
    verts = Array{Float64}(hcat(verts...)')
    return verts
end

function save_vertices(experiment::Experiments.Experiment; location = "verts.npy")
    verts = get_vertices(experiment)
    npzwrite(location, verts)
end

function simplex(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    K = size(A, 2)

    V, w = get_affine_isom(K)

    # Transform original constraints
    A_tilde = A * V
    b_tilde = b .- A * w
    
    # enforce nonnegativity
    A_new = vcat(A_tilde, -V)  # (m+K)×(K-1)
    b_new = vcat(b_tilde,  w)  # (m+K)

    return A_new, b_new
end

function get_affine_isom(K)
    # Create offset vector w as simplex centroid
    w = fill(1.0 / K, K)

    # Build basis matrix V
    V = vcat(
      Matrix{Float64}(I, K-1, K-1),   # (K-1)×(K-1)
      -ones(1, K-1)                   #  1×(K-1)
    )

    return V, w
end

function little_psi(verts)
    K = size(verts, 2) + 1

    # Get isomorphism
    V, w = get_affine_isom(K)

    # Invert
    simplex_coords = w .+ V * permutedims(verts, (2,1))

    return permutedims(simplex_coords, (2, 1))
end

function psi(X)
    n, N, K = size(X)

    # Get isomorphism
    V, w = get_affine_isom(K + 1)
    
    # Reshape X and apply transformation
    X_flat = reshape(X, n * N, K)
    T_flat = X_flat * permutedims(V, (2, 1))

    # Reshape T and apply affine offset
    T = reshape(T_flat, n, N, K + 1)
    T_aff = TensorOps.expand(repeat(w, 1, N)') .+ T
    return T_aff
end


end # module