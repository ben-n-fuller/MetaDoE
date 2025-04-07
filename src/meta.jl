module MetaDoE

include("doe/models.jl")
include("doe/designs.jl")
include("optim/objectives.jl")
include("util/tensor_ops.jl")
include("util/distributed_jobs.jl")

using .DistributedJobs
using .TensorOps
using .Designs
using .Models
using .Objectives

end # module