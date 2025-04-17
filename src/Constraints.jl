module Constraints

using LinearAlgebra
using Polyhedra
using HiGHS
using LinearAlgebra
using ..Experiments

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

function barycentric_embed(experiment::Experiments.Experiment)
    verts = get_vertices(experiment)
    n = size(verts, 2)
    corners = vcat(I(n-1), zeros(1, n-1))
    return verts * corners
end

function reparameterize_simplex(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    m, K = size(A)

    # Create offset vector w
    w = fill(1.0/K, K)

    # Build basis matrix V
    V = vcat(
      Matrix{Float64}(I, K-1, K-1),   # (K-1)×(K-1)
      -ones(1, K-1)                   #   1  ×(K-1)
    )

    # Transform original constraints
    A_tilde = A * V
    b_tilde = b .- A * w
    
    # enforce nonnegativity
    A_new = vcat(A_tilde, -V)  # (m+K)×(K-1)
    b_new = vcat(b_tilde,  w)  # (m+K)

    return A_new, b_new
end

end # module