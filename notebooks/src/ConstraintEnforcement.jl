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

function make_enforcer(enforcer::ConstraintEnforcer)::Function
    @match enforcer begin 
        ResampleEnforcer(constraints, initializer) =>
            (X_prev, X_curr, t) -> resample_violating_rows!(X_curr, constraints, initializer)

        LinearEnforcer(constraints) =>
            (X_prev, X_curr, t) -> repair_linear_intersect!(X_curr, X_prev, constraints)

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

    return constraints.λ * total_violation_per_particle
end

function resample_violating_rows!(X, constraints::LinearConstraints, initializer::Function)
    n, N, K = size(X)

    # Compute constraint violations
    clamped = compute_constraint_violations(X, constraints)

    # Identify violating particles
    is_violating_flat = vec(any(clamped .> 0.0, dims=1))  # shape: (nN,)
    violation_mask = reshape(is_violating_flat, n, N)
    violating_particles = any(violation_mask, dims=2) |> vec  # shape: (p,)

    p = sum(violating_particles)

    # Re-sample
    X[violating_particles, :, :] .= initializer(p)

    return p
end

function linear_intersection(A, b, x_int, x_ext)

    # Identify which rows of A are violated
    violated = (A * x_ext) .> b

    # If none violated, return x_ext
    if !any(violated)
        return x_ext
    end

    # Solve for lambda for each of the violated constraints
    A_violated = A[violated, :]
    b_violated = b[violated]
    numerator = b_violated .- (A_violated * x_int)
    denominator = A_violated * (x_ext .- x_int)

    # If denominator has any zero, handle or skip accordingly
    λ_vec = numerator ./ denominator

    # Find the minimizing lambda
    λ_min = minimum(λ_vec)

    # Return the intersection point
    return x_int .+ λ_min .* (x_ext .- x_int)
end

function repair_linear_intersect!(X_int, X_ext, constraints::LinearConstraints)
    A = constraints.A
    b = constraints.b 

    n, N, K = size(X_int)

    # Flatten to (n*N, K) so each design point is a row
    X_int_2d = reshape(X_int, n*N, K)
    X_ext_2d = reshape(X_ext, n*N, K)

    # Identify violating design points
    violation_mat = A * X_ext_2d' .- b
    clamped = max.(violation_mat, 0)

    # Create a mask for violating rows
    is_violating = vec(any(clamped .> 0, dims=1))

    # Loop over violating rows
    violating_indices = findall(is_violating)
    for i in violating_indices
        # Extract the row
        x_int_row = @view X_int_2d[i, :]
        x_ext_row = @view X_ext_2d[i, :]

        # Repair that row
        x_repaired = linear_intersection(A, b, x_int_row, x_ext_row)

        # In-place update
        x_ext_row .= x_repaired
    end

    X_ext .= reshape(X_ext_2d, n, N, K)
    return sum(is_violating)
end

end # module