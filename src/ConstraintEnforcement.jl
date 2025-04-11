module ConstraintEnforcement

using MLStyle

@data Constraints begin
    LinearConstraints(A::Array{Float64, 2}, b::Vector{Float64})
    NoConstraints()
end

struct LinearConstraints <: Constraints
    A::Array{Float64, 2}
    b::Vector{Float64}
end

struct NoConstraints <: Constraints end

@data ConstraintEnforcer begin
    ResampleEnforcer(constraints::LinearConstraints, initializer::Function)
    LinearEnforcer(constraints::LinearConstraints)
    PenaltyEnforcer(constraints::LinearConstraints)
end

struct ResampleEnforcer <: ConstraintEnforcer
    constraints::LinearConstraints
    initializer::Function
end

struct LinearEnforcer <: ConstraintEnforcer 
    constraints::LinearConstraints  
end

struct PenaltyEnforcer <: ConstraintEnforcer 
    constraints::LinearConstraints
end

@enum EnforcerType Parametric Penalty Resample

function make_enforcer_func(enforcer::ConstraintEnforcer)::Function
    @match enforcer begin 
        ResampleEnforcer(constraints, initializer) =>
            (X_prev, X_curr, t) -> resample_violating_rows(X_curr, constraints, initializer)

        LinearEnforcer(constraints) =>
            (X_prev, X_curr, t) -> repair_linear_intersect(X_prev, X_curr, constraints)

        PenaltyEnforcer(constraints) =>
            (X_prev, X_curr, t) -> apply_penalty(X_curr, constraints)

        _ => error("Unsupported enforcer type: $(typeof(enforcer))")
    end
end

function compute_constraint_violations(X, constraints::LinearConstraints)
    n, N, K = size(X)

    # Reshape X to ((n*N), K).
    X_2d = reshape(X, n*N, K)

    # Compute violations
    violation_mat = (constraints.A * X_2d') .- constraints.b

    # Max(violation, 0)
    clamped = max.(violation_mat, 0)

    return clamped
end

function apply_penalty(X, constraints::LinearConstraints) 
    n, N, K = size(X)
    m = size(constraints.A, 1)

    clamped = compute_constraint_violations(X, constraints)

    # Sum the squared violation over the m constraints  
    violation_sums_1d = vec(sum(abs2.(clamped), dims=1))

    # Reshape
    violation_sums_2d = reshape(violation_sums_1d, n, N)

    # Sum across all design points per candidate
    total_violation_per_particle = vec(sum(violation_sums_2d, dims=2)) ./ (N * m)

    return 0.5 * total_violation_per_particle
end

function resample_violating_rows(X, constraints::LinearConstraints, initializer::Function)
    n, N, K = size(X)

    # Compute constraint violations
    clamped = compute_constraint_violations(X, constraints)

    # Identify violating particles
    is_violating_flat = vec(any(clamped .> 0.0, dims=1))  # shape: (nN,)
    violation_mask = reshape(is_violating_flat, n, N)
    violating_particles = any(violation_mask, dims=2) |> vec  # shape: (p,)

    p = sum(violating_particles)

    X_res = copy(X)

    # Re-sample
    X_res[violating_particles, :, :] .= initializer(p)

    return X_res
end

function repair_linear_intersect(
    X_int::Array{Float64,3},
    X_ext::Array{Float64,3},
    constraints::LinearConstraints
)
    A = constraints.A   # shape (m, K)
    b = constraints.b   # shape (m)
    
    # Flatten and transpose input to simplify constraint computations
    # Now each row is a design point
    n, N, K = size(X_int)
    X_int_flat = reshape(X_int, n*N, K)'  # (K, n*N)
    X_ext_flat = reshape(X_ext, n*N, K)'  # (K, n*N)

    # Identify violating design points
    violation_mat  = A * X_ext_flat .- b # (m, n*N)
    violation_mask = violation_mat .> 0 # (m, n*N)

    # Shortcut if no violations
    if !any(violation_mask)
        return copy(X_ext)
    end

    # Compute projection
    # numerator[i,j]   = b[i] - A[i,:] * X_int_j
    # denominator[i,j] = A[i,:] * (X_ext_j - X_int_j)
    numerator   = b .- (A * X_int_flat)     # (m, n*N)
    direction = X_ext_flat .- X_int_flat    # (K, n*N)
    denominator = A * direction             # (m, n*N)

    # Solve for λ
    # For non-violating rows, assign Inf
    λ_all = fill(Inf, size(numerator))  # shape: (m, n*N)
    λ_all[violation_mask] .= numerator[violation_mask] ./ denominator[violation_mask]


    # Find the min λ across all constraints
    λ_min_per_row = minimum(λ_all, dims=1)  # shape (1, n*N)

    # Any non-violating rows will be set to Inf
    # Leave these in their current position
    is_violating_row = vec(any(violation_mask, dims=1))  # (n*N)
    λ_min_per_row[1, .!is_violating_row] .= 1.0

    # Repair the violating points by interpolating with λ_min
    X_repaired_flat = X_int_flat .+ direction .* λ_min_per_row

    # Reshape to original dimensions
    X_repaired_2d = X_repaired_flat'                    # (n*N, K)
    X_repaired = reshape(X_repaired_2d, n, N, K)        # (n, N, K)

    return X_repaired
end


end # module