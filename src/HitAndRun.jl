module HitAndRun 

using Polyhedra
using LinearAlgebra
using Random

function sample_random_direction(num_samples, n)
    v = randn(num_samples, n)
    return v ./ norm.(eachrow(v))
end

# Provides bounds for t in a parameterized line v + td inside the polytope defined by Ax <= b
# A is m×n, b is m×1, v is n-element vector (current point),
# d is n-element direction vector.
# Returns (t_min, t_max) so that for all i, A_i*(v + t*d) <= b_i.
function get_bounds(A, b, v, d)
    unscaled = (b .- (A * v))[:, 1]
    scale_factors = (A * d)[:, 1]

    pos_mask = scale_factors .> 0
    neg_mask = scale_factors .< 0
    zero_mask = scale_factors .== 0

    if any(zero_mask .& (unscaled .< 0))
        return (NaN, NaN)
    end

    t_max = Inf
    if any(pos_mask)
        t_max = minimum(unscaled[pos_mask] ./ scale_factors[pos_mask])
    end

    t_min = -Inf
    if any(neg_mask)
        t_min = maximum(unscaled[neg_mask] ./ scale_factors[neg_mask])
    end

    return (t_min, t_max)
end


function hit_and_run_start(A, b, v, n, rng)
    n_size = length(v)
    samples = zeros(n, n_size)

    sample_count = 0
    while sample_count < n
        # Sample from the unit hypersphere
        d = randn(n_size)
        d /= norm(d)

        # Get bounds for t
        t_min, t_max = get_bounds(A, b, v, d)

        # Skip if bounds are invalid or infeasible
        if isnan(t_min) || isnan(t_max) || t_min > t_max || isinf(t_min) || isinf(t_max)
            continue  # Try a new direction
        end

        # Sample t uniformly in [t_min, t_max]
        t = rand(rng) * (t_max - t_min) + t_min
        v = v + t * d
        sample_count += 1
        samples[sample_count, :] = v
    end

    return samples
end

function get_initial_sample(A, b, lib)
    p = polyhedron(hrep(A, b), lib)
    center, radius = chebyshevcenter(p)
    return center
end

function hit_and_run(A, b, n, lib; burnin=100, rng = Random.GLOBAL_RNG, v_0 = nothing)
    # Obtain initial interior point
    if v_0 === nothing
        v_0 = get_initial_sample(A, b, lib)
    end

    # Run burn-in phase
    burnin_samples = hit_and_run_start(A, b, v_0, burnin, rng)

    # Produce samples
    v = burnin_samples[end, :]
    return hit_and_run_start(A, b, v, n, rng)
end


end # module