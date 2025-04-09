module Constraints

using LinearAlgebra

function unit_hypercube_constraints(d::Int)
    I_d = Matrix{Float64}(I, d, d)
    A = vcat(I_d, -I_d)
    b = vcat(ones(d), zeros(d))
    return A, b
end

function hypercube_constraints(n::Int, k::Real)
    I_n = Matrix{Float64}(I, n, n)
    A = vcat(I_n, -I_n)
    b = vcat(fill(Float64(k), n), fill(Float64(k), n))
    return A, b
end

end # module