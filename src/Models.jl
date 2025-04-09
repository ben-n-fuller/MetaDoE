module Models

using Combinatorics
using LinearAlgebra

function expand(X; left=true, n=3)
    if ndims(X) < n
        shape = left ? (ones(Int, n - ndims(X))..., size(X)...) : (size(X)..., ones(Int, n - ndims(X))...)
        return reshape(X, shape...)
    end
    X
end

function squeeze(X)
    singleton_dims = findall(size(X) .== 1)
    dropdims(X, dims=tuple(singleton_dims...)) 
end

function factory_base(funcs::Vector, data_selector::Function; squeeze_output=true)
    builder = (X) -> cat([(f ∘ data_selector ∘ expand)(X) for f in funcs]..., dims=3)
    if squeeze_output
        return squeeze ∘ builder
    end

    builder
end

function create(funcs::Vector; squeeze_output=true)
    # Function to expose columns of tensor
    g = (X) -> (i) -> X[:, :, i]
    factory_base(funcs, g, squeeze_output=squeeze_output)
end

function build_full(order)
    idx_combinations = collect(combinations(1:(order+1), 2))
    func = (x, i, j) -> (x(i) .* x(j)) .* (x(i) .- x(j))
    (x) -> cat([func(x, i, j) for (i,j) in idx_combinations]..., dims=3)
end

function create(;order=1, interaction_order=0, include_intercept=true, transforms=[], full=false, squeeze_output=true)
    function model_builder(X)
        # List of funcs to define model
        funcs = []

        # Expand terms 
        X = expand(X)

        # Intercept
        if include_intercept
            push!(funcs, intercept())
        end

        if full
            push!(funcs, build_full(interaction_order))
        end

        # Power terms, including main effects
        if order > 0
            push!(funcs, powers(1:size(X, 3), 1:order))
        end

        # Interactions starting with order 2
        if interaction_order > 0
            push!(funcs, interactions(1:size(X, 3), 2:(interaction_order+1)))
        end

        # Custom functions
        push!(funcs, transforms...)

        # Get builder from base factory
        create(funcs, squeeze_output = squeeze_output)(X)
    end
end

function powers(factors, powers)
    (x) -> cat([x(i) .^ j for i in factors for j in powers]..., dims=3)
end

function interactions(factors, orders)
    # Find interaction terms for each order and flatten
    ints = vcat([collect(combinations(factors, o)) for o in orders]...)

    # Build interaction terms
    interactions(ints)
end

function interactions(ints::Vector)
    build_interaction = (x, combo) -> reduce(.*, [x(i) for i in combo])
    (x) -> cat([expand(build_interaction(x, combo), left=false) for combo in ints]..., dims=3)
end


function intercept()
    (x) -> ones(size(x(1), 1), size(x(1), 2), 1)
end

function add(model_builder::Function, transforms::Vector)
    (X) -> cat(model_builder(X), create(transforms)(X), dims=3)
end

function add(model_builder::Function, transform::Function)
    add(model_builder, [transform])
end

####################
# Export functions #
####################

# Factory method for building model matrices and add method to augment model matrices
export create, add

# Common model construction functions
export intercept, powers, interactions

# Utility
export expand, squeeze

# Functions for building default model matrices
linear = create()
quadratic = create(order=2)
cubic = create(order=3)
linear_interaction = create(interaction_order=1)
quadratic_interaction = create(order=2, interaction_order=1)
cubic_interaction = create(order=3, interaction_order=1)
builder = create(include_intercept=false, order=0)
scheffe = (n) -> create(interaction_order=(n-1), include_intercept=false)
special_cubic = scheffe(3)
full_cubic = create(order=1, interaction_order=2, include_intercept=false, full=true)
full_quartic = create(order=1, interaction_order=3, include_intercept=false, full=true)

export linear, quadratic, cubic, linear_interaction, quadratic_interaction, cubic_interaction, scheffe, full_cubic, builder

# End module
end
