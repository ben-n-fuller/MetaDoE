module TensorOps

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

export squeeze, expand

end