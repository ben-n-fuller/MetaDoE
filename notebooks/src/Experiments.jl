module Experiments

using ..ConstraintEnforcement

using Random
using MLStyle

struct Experiment
    factors::Dict{String, Int64}
    constraints::ConstraintEnforcement.Constraints
    N::Int64 
    K::Int64 
end

function create(factors::Vector{String}, N::Int64, K::Int64)
    factor_dict = Dict(f => i for (i, f) in enumerate(factors))
    return Experiment(factor_dict, ConstraintEnforcement.NoConstraints(), N, K)
end

function create(N::Int64, K::Int64)
    factors = Dict(string(i) => i for i in 1:K)
    constraints = ConstraintEnforcement.NoConstraints()
    Experiment(factors, constraints, N, K)
end

function with_interval_constraint(exp::Experiment, factor::String, lower::Float64, upper::Float64)
    factor_index = exp.factors[factor]
    with_interval_constraint(exp, factor_index, lower, upper)
end

function with_factor_ratio(exp::Experiment, factor_1::String, factor_2::String, ratio::Float64)
    factor1_index = exp.factors[factor_1]
    factor2_index = exp.factors[factor_2]
    with_factor_ratio(exp, factor1_index, factor2_index, ratio)
end

function with_interval_constraint(exp::Experiment, factor_index::Int64, lower::Float64, upper::Float64)
    lower_bound_row = zeros(exp.K)
    upper_bound_row = zeros(exp.K)
    lower_bound_row[factor_index] = 1
    upper_bound_row[factor_index] = -1

    @match exp.constraints begin
        ConstraintEnforcement.LinearConstraints(A, b) => begin
            new_A = vcat(A, lower_bound_row', upper_bound_row')
            new_b = vcat(b, lower, -upper)
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        ConstraintEnforcement.NoConstraints() => begin
            new_A = vcat(lower_bound_row', upper_bound_row')
            new_b = [lower, -upper]
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        _ => error("Unsupported constraint type: $(typeof(exp.constraints))")
    end
end

function with_factor_ratio(exp::Experiment, factor1_index::Int64, factor2_index::Int64, ratio::Float64)
    row = zeros(exp.K)
    row[factor1_index] = 1
    row[factor2_index] = -ratio

    @match exp.constraints begin
    ConstraintEnforcement.LinearConstraints(A, b) => begin
            new_A = vcat(A, row')
            new_b = vcat(b, 0.0)
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        ConstraintEnforcement.NoConstraints() => begin
            new_A = row'
            new_b = [0.0]
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        _ => error("Unsupported constraint type: $(typeof(exp.constraints))")
    end
end

function with_linear_constraint(exp::Experiment, constraint::Vector{Float64}, bound::Float64)
    @assert length(constraint) == exp.K "Constraint vector must have length equal to number of factors (K = $(exp.K))"

    @match exp.constraints begin
        ConstraintEnforcement.LinearConstraints(A, b) => begin
            new_A = vcat(A, constraint')
            new_b = vcat(b, bound)
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        ConstraintEnforcement.NoConstraints() => begin
            new_A = constraint'
            new_b = [bound]
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        _ => error("Unsupported constraint type: $(typeof(exp.constraints))")
    end
end

function with_linear_constraints(exp::Experiment, constraint::Array{Float64, 2}, bound::Vector{Float64})
    @assert size(constraint, 2) == exp.K "Constraint matrix must have K columns (K = $(exp.K))"
    @assert size(constraint, 1) == length(bound) "Number of rows in constraint matrix must match length of bound vector"

    @match exp.constraints begin
        ConstraintEnforcement.LinearConstraints(A, b) => begin
            new_A = vcat(A, constraint)
            new_b = vcat(b, bound)
            new_constraints = ConstraintEnforcement.LinearConstraints(new_A, new_b)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        ConstraintEnforcement.NoConstraints() => begin
            new_constraints = ConstraintEnforcement.LinearConstraints(constraint, bound)
            Experiment(exp.factors, new_constraints, exp.N, exp.K)
        end

        _ => error("Unsupported constraint type: $(typeof(exp.constraints))")
    end
end

end # module