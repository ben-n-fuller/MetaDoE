module IndustrialStats

include("doe/models.jl")
include("doe/designs.jl")
include("doe/objectives.jl")
include("util/tensor_ops.jl")
include("util/distributed_jobs.jl")
include("pso/particle_swarm.jl")
include("geom/simplex.jl")

using .DistributedJobs
using .TensorOps
using .Designs
using .Models
using .Objectives
using .ParticleSwarm
using .Simplex

end # module