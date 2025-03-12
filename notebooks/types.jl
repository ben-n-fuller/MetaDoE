module PSOTypes

export HyperParams, ParticleMemory, ParticleFitness, ParticleState, Swarm, RunnerState, RunnerParams

# State update hyperparameters
struct HyperParams 
    num_particles::Int # Number of particles
    w::Float64 # Inertia weight
    c1::Float64 # Cognitive weight
    c2::Float64 # Social weight
    num_neighbors::Int # Avg. number of neighbors per particle
end

# Memory per particle and global bests
struct ParticleMemory
    particle_best::Array{Float64, 3}
    particle_best_scores::Vector
    global_best::Array{Float64, 2}
    global_best_score::Float64
end

# Current fitness scores for each particle w.r.t. some objective function
struct ParticleFitness
    scores::Vector
    objective::Function
end

# Current position and velocity of each particle
struct ParticleState
    positions::Array{Float64, 3}
    velocities::Array{Float64, 3}
end

# Global state of the particle swarm
struct Swarm
    state::ParticleState
    neighbors
    fitness::ParticleFitness  
    memory::ParticleMemory
    params::HyperParams
    constraints::Function
end

# State of the optimization runner
struct RunnerState
    swarm::Swarm
    iter::Int
    stagnation::Int
end

# Parameters for the optimization runner
struct RunnerParams
    max_iter::Int
    max_stag::Int
    rel_tol::Float64
end

end