module Models

using Combinatorics
using LinearAlgebra

"""
    expand(X; left=true, n=3)

Expand the dimensions of an array `X` by adding singleton dimensions.

Arguments:
- `X`: The input array.
- `left`: Boolean indicating whether to add singleton dimensions on the left (default: `true`).
- `n`: The desired number of dimensions for the output array (default: `3`).

Returns:
The input array `X` with additional singleton dimensions added.
"""
function expand(X; left=true, n=3)
    if ndims(X) < n
        shape = left ? (ones(Int, n - ndims(X))..., size(X)...) : (size(X)..., ones(Int, n - ndims(X))...)
        return reshape(X, shape...)
    end
    X
end

"""
    squeeze(X)

Remove singleton dimensions from an array `X`.

# Arguments
- `X`: The input array.

# Returns
- The input array `X` with singleton dimensions removed.
"""
function squeeze(X)
    singleton_dims = findall(size(X) .== 1)
    dropdims(X, dims=tuple(singleton_dims...)) 
end

"""
    factory_base(funcs::Vector, data_selector::Function)

Constructs a builder function that applies a series of input functions to a tensor along the factor pages.

# Arguments
- `funcs::Vector`: A vector of functions to be applied to columns and possibly combinations of columns along the third dimension of the tensor.
- `data_selector::Function`: A function that specifies which data is exposed to the input functions.

# Returns
A builder function that applies the input functions column-wise to the tensor.
"""
function factory_base(funcs::Vector, data_selector::Function; squeeze_output=true)
    builder = (X) -> cat([(f ∘ data_selector ∘ expand)(X) for f in funcs]..., dims=3)
    if squeeze_output
        return squeeze ∘ builder
    end

    builder
end

"""
    create(funcs::Vector)

Create a factory function that takes a vector of functions and returns a model builder.
The returned function takes a tensor `X` and an index `i`, and returns the `i`-th column of `X`.

# Arguments
- `funcs::Vector`: A vector of functions.

# Returns
A new function that takes a tensor `X` and an index `i`, and returns the `i`-th column of `X`.
"""
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

"""
    create(; order=1, interaction_order=1, intercept=true, transforms=[])

Constructs a model builder function that generates a model based on the specified parameters.

## Arguments
- `order`: The order of the power terms, including main effects. Default is 1.
- `interaction_order`: The order of the interaction terms. Default is 1.
- `intercept`: Whether to include an intercept term in the model. Default is `true`.
- `transforms`: A list of custom functions to be applied to the model.

## Returns
A model builder function that takes a tensor `X` as input and returns a model matrix computed from X.

"""
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


"""
    powers(factors, powers)

Build power terms for a given set of factors and powers.

# Arguments
- `factors`: An array of factors, representing the factors to which the power terms are applied.
- `powers`: An array of powers, representing the powers to which each factor in the factors vector is raised.

# Returns
An array of power terms.
"""
function powers(factors, powers)
    (x) -> cat([x(i) .^ j for i in factors for j in powers]..., dims=3)
end

"""
    interactions(factors, orders)

Builds interaction terms for a given set of factors and orders.

# Arguments
- `factors`: An array of factors. All combinations of length order for order in orders will be used to build interaction terms.
- `orders`: An array of orders.

# Returns
An array of interaction terms.
"""
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

"""
    intercept()

Create an intercept function that returns a function that generates an array of ones with the same size as the input.

# Arguments
- `x`: A function that takes an input and returns an array.

# Returns
A function that takes an input and returns an array of ones with the same size as the input.

"""
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
