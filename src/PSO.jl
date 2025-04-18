module PSO

using ..TensorOps
using ..ConstraintEnforcement
using ..Experiments
using ..Designs

using Distributions
using LinearAlgebra
using SparseArrays
using MLStyle
using Random
using NPZ

# State update hyperparameters
struct HyperParams 
    num_particles::Int # Number of particles
    w::Float64         # Inertia weight
    c1::Float64        # Cognitive weight
    c2::Float64        # Social weight
    num_neighbors::Int # Avg. number of neighbors per particle
end

# Memory per particle and global bests
struct ParticleMemory
    previous_positions::Array{Float64, 3}  # Previous location of all particles
    particle_best::Array{Float64, 3}       # Best position for each particle
    particle_best_scores::Vector           # Best score for each particle
    global_best::Array{Float64, 2}         # Best position across all particles
    global_best_score::Float64             # Best score across all particles
end

# Current fitness scores for each particle w.r.t. some objective function
struct ParticleFitness
    scores::Vector
end

# Current position and velocity of each particle
struct ParticleState
    positions::Array{Float64, 3}
    velocities::Array{Float64, 3}
end

struct Objective
    enforcer::Function
    objective::Function
end

# Global state of the particle swarm
struct Swarm
    state::ParticleState
    neighbors::SparseArrays.SparseMatrixCSC{Bool, Int64}
    fitness::ParticleFitness  
    memory::ParticleMemory
    params::HyperParams
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

struct OptimizationContext 
    initial_world::Swarm
    runner_params::RunnerParams 
    callback::Function 
    check_convergence::Function
    updater::Function 
    improved::Function 
    objective::Objective
end

function update_memory(memory::ParticleMemory, fitness::ParticleFitness, state::ParticleState, old_state::ParticleState)::ParticleMemory
    # Update best positions
    memory.particle_best[fitness.scores .< memory.particle_best_scores, :, :] .= state.positions[fitness.scores .< memory.particle_best_scores, :, :]

    # Update best scores
    memory.particle_best_scores[fitness.scores .< memory.particle_best_scores] .= fitness.scores[fitness.scores .< memory.particle_best_scores]

    # Update global best score and position
    new_global_best = memory.particle_best[argmin(memory.particle_best_scores), :, :]

    new_best_score = minimum(memory.particle_best_scores)

    return ParticleMemory(old_state.positions, memory.particle_best, memory.particle_best_scores, new_global_best, new_best_score)
end

function update_fitness(state::ParticleState, obj::Function, t::Int64)::ParticleFitness
    # Compute scores from updated positions
    scores = obj(state.positions, state.positions, t)

    # Update fitness
    return ParticleFitness(scores)
end

function global_best_social(world::Swarm)
    r2 = rand(size(world.state.positions)...)
    return world.params.c2 * r2 .* TensorOps.expand(world.memory.global_best) .- world.state.positions
end

function neighborhood_best_social(world::Swarm)
    r2 = rand(size(world.state.positions)...)
    return world.params.c2 * r2 .* TensorOps.expand(world.memory.particle_best[:, :, world.neighbors]) .- world.state.positions
end

function particle_best_cognitive(world::Swarm)
    r1 = rand(size(world.state.positions)...)
    return world.params.c1 * r1 .* (world.memory.particle_best .- world.state.positions)
end

function default_inertia(world::Swarm)
    return world.params.w * world.state.velocities
end

function update_vel_pos(world::Swarm, objective::Objective, social::Function, cognitive::Function, inertia::Function, t::Int64)::ParticleState
    # Update velocities with inertia, cognitive and social components
    particle_inertia = inertia(world)
    cognitive = cognitive(world)
    social = social(world)

    # Set new velocity
    new_velocities = particle_inertia .+ cognitive .+ social

    # Update particle positions
    new_particle_pos = world.state.positions .+ new_velocities

    # Apply constraints
    constrained_positions, constrained_velocities = objective.enforcer(world.state.positions, new_particle_pos, new_velocities, t)

    return ParticleState(constrained_positions, constrained_velocities)
end

function create_adjacency_matrix(num_particles, num_neighbors)
    # Create a boolean adjacency matrix of particles
    # Each neighbor has on average num_neighbors connections
    d = Uniform(0, 1)
    p_avg = 1 - (1 - (1 / num_particles)) ^ num_neighbors
    adjacency_matrix = reshape(rand(d, num_particles * num_particles) .< p_avg, (num_particles, num_particles))

    # Connect particles to themselves
    adjacency_matrix[diagind(adjacency_matrix)] .= 1

    return sparse(adjacency_matrix)
end

function update_neighbors(world::Swarm)::Swarm
    neighbs = create_adjacency_matrix(swarm.params.num_particles, swarm.params.num_neighbors)
    return Swarm(world.state, neighbs, world.fitness, world.memory, world.params)
end

function update_swarm(world::Swarm, state::ParticleState, fitness::ParticleFitness, memory::ParticleMemory)::Swarm
    return Swarm(state, world.neighbors, fitness, memory, world.params)
end

function update_state_default(world::Swarm, objective::Objective, stagnant::Bool, t::Int)::Swarm
    new_state = update_vel_pos(world, objective, global_best_social, particle_best_cognitive, default_inertia, t)
    new_fitness = update_fitness(new_state, objective.objective, t)
    new_memory = update_memory(world.memory, new_fitness, new_state, world.state)
    new_world = update_swarm(world, new_state, new_fitness, new_memory)
    if stagnant
        new_world = update_neighbors(new_world)
    end
    return new_world
end

function update_state_neighbors(world::Swarm, objective::Objective, stagnant::Bool, t::Int)::Swarm
    new_state = update_vel_pos_neighbors(world)
    new_fitness = update_fitness(new_state, objective.objective, t)
    new_memory = update_memory(world.memory, new_fitness, new_state, world.state)
    new_world = update_swarm(world, new_state, new_fitness, new_memory)
    if stagnant
        new_world = update_neighbors(new_world)
    end
    return new_world
end

function initialize_swarm(initializer::Function, objective::Objective, params::HyperParams)::Swarm
    particles = initializer(params.num_particles)
    velocities = initializer(params.num_particles)
    particle_state = ParticleState(particles, velocities)
    neighbors = create_adjacency_matrix(params.num_particles, params.num_neighbors)
    scores = objective.objective(particles, particles, 0)
    memory = ParticleMemory(particles, particles, scores, particles[argmin(scores), :, :], minimum(scores))
    fitness = ParticleFitness(scores)
    return Swarm(
        particle_state, 
        neighbors, 
        fitness,
        memory,
        params
    )
end


function get_enforcer(enforcer_type::ConstraintEnforcement.EnforcerType, experiment::Experiments.Experiment, initializer::Function)
    if enforcer_type == ConstraintEnforcement.Parametric
        return  ConstraintEnforcement.LinearEnforcer(experiment.constraints)
    elseif enforcer_type == ConstraintEnforcement.Resample 
        return ConstraintEnforcement.ResampleEnforcer(experiment.constraints, initializer)
    end

    return ConstraintEnforcement.PenaltyEnforcer(experiment.constraints)
end

function create_context(
    experiment::Experiments.Experiment, 
    obj::Function; 
    hyperparams = default_hyperparams(), 
    runner_params = default_runner_params(),
    callback = default_logger(),
    rng = Random.GLOBAL_RNG,
    enforcer_type = ConstraintEnforcement.Parametric,
    use_model = true)
    initializer = Designs.create_initializer(experiment.constraints, experiment.N, experiment.K; rng = rng)
    enforcer = get_enforcer(enforcer_type, experiment, initializer)
    objective = use_model ? obj ∘ experiment.model : obj
    new_objective = PSO.create_objective(objective, enforcer)
    swarm = PSO.initialize_swarm(initializer, new_objective, hyperparams)
    return OptimizationContext(
        swarm, 
        runner_params, 
        callback, 
        max_iters_or_stagnation, 
        update_state_default, 
        global_best_improvement, 
        new_objective
    )
end

function max_iters_or_stagnation(runner::RunnerState, params::RunnerParams)
    return runner.iter < params.max_iter && runner.stagnation < params.max_stag
end

function global_best_improvement(old_world::Swarm, new_world::Swarm, runner_params::RunnerParams)
    return abs(new_world.memory.global_best_score - old_world.memory.global_best_score) > runner_params.rel_tol
end

function optimize(context::OptimizationContext)
    runner_state = RunnerState(context.initial_world, 0, 0)
    res = context.callback(runner_state)
    while context.check_convergence(runner_state, context.runner_params)
        new_world = context.updater(runner_state.swarm, context.objective, runner_state.stagnation >= context.runner_params.max_stag, runner_state.iter)
        currently_improved = context.improved(runner_state.swarm, new_world, context.runner_params)
        stagnation = currently_improved ? 0 : runner_state.stagnation + 1
        runner_state = RunnerState(new_world, runner_state.iter + 1, stagnation)
        res = context.callback(runner_state)
    end
    return runner_state, res
end

function create_hyperparams(S, w, c1, c2, num_neighbors)::HyperParams
    return HyperParams(S, w, c1, c2, num_neighbors)
end

function create_hyperparams(S::Int64)::HyperParams
    return create_hyperparams(S, 1/(2*log(2)), 0.5 + log(2), 0.5 + log(2), 3)
end

@enum EnforcerType LinearIntersection Resample Penalty
function get_enforcer(exp::Experiments.Experiment, enforcer_type::EnforcerType)
    @match enforcer_type begin 
        LinearIntersection => ConstraintEnforcement.LinearEnforcer(exp.constraints)
        Resample => ConstraintEnforcement.ResampleEnforcer(exp.constraints, Experiments.get_initializer(exp))
        Penalty => ConstraintEnforcement.PenaltyEnforcer(exp.constraints)
        _ => error("Unsupported constraint type: $(typeof(exp.constraints))")
    end
end

function create_objective(obj::Function)::Objective
    return Objective((X_prev, X_curr, velocity, t) -> (X_curr, velocity), (X_prev, X_curr, t) -> obj(X_curr))
end

function get_penalty_enforcer(obj::Function, constraints::ConstraintEnforcement.ConstraintEnforcer)
    enforcer_func = ConstraintEnforcement.make_enforcer_func(constraints)
    Objective((X_prev, X_curr, velocity, t) -> (X_curr, velocity), (X_prev, X_curr, velocity, t) -> obj(X_prev) .+ enforcer_func(X_prev, X_curr, t))
end

function create_objective(obj::Function, constraints::ConstraintEnforcement.ConstraintEnforcer)::Objective
    @match constraints begin
        ConstraintEnforcement.PenaltyEnforcer(linear_constraints) => get_penalty_enforcer(obj, constraints)
        ConstraintEnforcement.ResampleEnforcer(linear_constraints, initializer) => Objective(ConstraintEnforcement.make_enforcer_func(constraints), (X_prev, X_curr, t) -> obj(X_curr))
        ConstraintEnforcement.LinearEnforcer(linear_constraints) => Objective(ConstraintEnforcement.make_enforcer_func(constraints), (X_prev, X_curr, t) -> obj(X_curr))
    end
end

function default_hyperparams()::HyperParams
    return create_hyperparams(100, 1/(2*log(2)), 0.5 + log(2), 0.5 + log(2), 3)
end


function default_runner_params()::RunnerParams
    return RunnerParams(500, 500, 1e-6)
end

function runner_params(max_iter, max_stag, rel_tol)
    return RunnerParams(max_iter, max_stag, rel_tol)
end

function default_logger()
    function logger(runner_state::RunnerState)
        println("Iteration: ", runner_state.iter, " Best score: ", runner_state.swarm.memory.global_best_score)
        return runner_state
    end
    return logger
end

function aggregate_results(;save_world=false)
    res = []
    function logger(runner_state::RunnerState)
        if save_world
            push!(res, runner_state.swarm)
        else
            push!(res, runner_state.swarm.memory.global_best_score)
        end
        return res
    end
    return logger
end

function get_optimizer(runner_state::RunnerState)
    return runner_state.swarm.memory.global_best, runner_state.swarm.memory.global_best_score
end

function save_history(history::Vector; location = "pso.npy")
    velocities = cat([h.state.velocities for h in history]...; dims=2)
    velocities = permutedims(velocities, (2, 1, 3))

    positions = cat([h.state.positions for h in history]...; dims=2)
    positions = permutedims(positions, (2, 1, 3))

    scores = cat([h.fitness.scores for h in history]...; dims=2)'
    npzwrite(location, Dict("positions" => positions, "velocities" => velocities, "scores" => scores))
end

function save_history_3d(history::Vector; location = "pso.npy")
    velocities = cat([h.state.velocities for h in history]...; dims=4)
    velocities = permutedims(velocities, (4, 1, 2, 3))

    positions = cat([h.state.positions for h in history]...; dims=4)
    positions = permutedims(positions, (4, 1, 2, 3))

    scores = cat([h.fitness.scores for h in history]...; dims=2)'
    npzwrite(location, Dict("positions" => positions, "velocities" => velocities, "scores" => scores))
end

function save_results(runner_state::RunnerState; location = "pso.npy")
    optimizer, optimum = get_optimizer(runner_state)
    npzwrite(location, 
        Dict(
            "optimizer" => optimizer,
            "optimum" => optimum
        )
    )
end

end # module