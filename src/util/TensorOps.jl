module TensorOps

function squeeze(X)
    singleton_dims = findall(size(X) .== 1)
    dropdims(X, dims=tuple(singleton_dims...)) 
end

function expand(X; left=true, n=3)
    if ndims(X) < n
        shape = left ? (ones(Int, n - ndims(X))..., size(X)...) : (size(X)..., ones(Int, n - ndims(X))...)
        return reshape(X, shape...)
    end
    X
end

export squeeze, expand

end