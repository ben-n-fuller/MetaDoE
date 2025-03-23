module PSO

include("./types.jl")
include("../src/util/tensor_ops.jl")
using .PSOTypes
using .TensorOps

using Distributions
using LinearAlgebra
using SparseArrays

function update_memory(memory::ParticleMemory, fitness::ParticleFitness, state::ParticleState)::ParticleMemory
    # Update best positions
    memory.particle_best[fitness.scores .< memory.particle_best_scores, :, :] .= state.positions[fitness.scores .< memory.particle_best_scores, :, :]

    # Update best scores
    memory.particle_best_scores[fitness.scores .< memory.particle_best_scores] .= fitness.scores[fitness.scores .< memory.particle_best_scores]

    # # Update global best score and position
    new_global_best = TensorOps.squeeze(memory.particle_best[argmin(memory.particle_best_scores), :, :])
    new_best_score = minimum(memory.particle_best_scores)

    return ParticleMemory(memory.particle_best, memory.particle_best_scores, new_global_best, new_best_score)
end

function update_fitness(state::ParticleState, objective::Function)::ParticleFitness
    # Compute scores from updated positions
    scores = objective(state.positions)

    # Update fitness
    return ParticleFitness(scores, objective)
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

function update_vel_pos(world::Swarm, social::Function, cognitive::Function, inertia::Function)::ParticleState
    # Update velocities with inertia, cognitive and social components
    particle_inertia = inertia(world)
    cognitive = cognitive(world)
    social = social(world)

    # Set new velocity
    new_velocities = particle_inertia .+ cognitive .+ social

    # Update particle positions
    new_particle_pos = world.state.positions .+ new_velocities

    # Apply constraints
    new_particle_pos = world.constraints(new_particle_pos)

    return ParticleState(new_particle_pos, new_velocities)
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
    return Swarm(world.state, neighbs, world.fitness, world.memory, world.params, world.constraints)
end

function update_swarm(world::Swarm, state::ParticleState, fitness::ParticleFitness, memory::ParticleMemory)::Swarm
    return Swarm(state, world.neighbors, fitness, memory, world.params, world.constraints)
end

function update_state_default(world::Swarm, stagnant::Bool, t::Int)::Swarm
    new_state = update_vel_pos(world, global_best_social, particle_best_cognitive, default_inertia)
    new_fitness = update_fitness(new_state, world.fitness.objective)
    new_memory = update_memory(world.memory, new_fitness, new_state)
    new_world = update_swarm(world, new_state, new_fitness, new_memory)
    if stagnant
        new_world = update_neighbors(new_world)
    end
    return new_world
end

function update_state_neighbors(world::Swarm, stagnant::Bool, t::Int)::Swarm
    new_state = update_vel_pos_neighbors(world)
    new_fitness = update_fitness(new_state, world.fitness.objective)
    new_memory = update_memory(world.memory, new_fitness, new_state)
    new_world = update_swarm(world, new_state, new_fitness, new_memory)
    if stagnant
        new_world = update_neighbors(new_world)
    end
    return new_world
end

function initialize_swarm(initializer::Function, objective::Function, constraints::Function, params::HyperParams)::Swarm
    particles = initializer(params.num_particles)
    velocities = initializer(params.num_particles)
    particle_state = ParticleState(particles, velocities)
    neighbors = create_adjacency_matrix(params.num_particles, params.num_neighbors)
    scores = objective(particles)
    memory = ParticleMemory(particles, scores, particles[argmin(scores), :, :], minimum(scores))
    fitness = ParticleFitness(scores, objective)
    return Swarm(
        particle_state, 
        neighbors, 
        fitness,
        memory,
        params,
        constraints
    )
end

function initialize_swarm(initializer::Function, objective::Function)::Swarm
    return initialize_swarm(initializer, objective, x -> x, default_hyperparams())
end

function initialize_runner(world::Swarm)
    return RunnerState(world, 0, 0)
end

function max_iters_or_stagnation(runner::RunnerState, params::RunnerParams)
    return runner.iter < params.max_iter && runner.stagnation < params.max_stag
end

function global_best_improvement(old_world::Swarm, new_world::Swarm, runner_params::RunnerParams)
    return abs(new_world.memory.global_best_score - old_world.memory.global_best_score) > runner_params.rel_tol
end

function optimize(world::Swarm, runner_params::RunnerParams, cb::Function, check_convergence::Function, updater::Function, improved::Function)
    runner_state = initialize_runner(world)
    res = cb(runner_state)
    while check_convergence(runner_state, runner_params)
        new_world = updater(runner_state.swarm, runner_state.stagnation >= runner_params.max_stag, runner_state.iter)
        currently_improved = improved(runner_state.swarm, new_world, runner_params)
        stagnation = currently_improved ? 0 : runner_state.stagnation + 1
        runner_state = RunnerState(new_world, runner_state.iter + 1, stagnation)
        res = cb(runner_state)
    end
    return runner_state, res
end

function optimize(world::Swarm, runner_params::RunnerParams, cb::Function)
    return optimize(world, runner_params, cb, max_iters_or_stagnation, update_state_default, global_best_improvement)
end

function optimize(world::Swarm, runner_params::RunnerParams)
    return optimize(world, runner_params, default_logger(), max_iters_or_stagnation, update_state_default, global_best_improvement)
end

function optimize(world::Swarm)
    return optimize(world, default_runner_params(), default_logger(), max_iters_or_stagnation, update_state_default, global_best_improvement)
end

function optimize(world::Swarm, cb::Function)
    return optimize(world, default_runner_params(), cb, max_iters_or_stagnation, update_state_default, global_best_improvement)
end

function create_hyperparams(S, w, c1, c2, num_neighbors)
    return HyperParams(S, w, c1, c2, num_neighbors)
end

function default_hyperparams()
    return create_hyperparams(100, 1/(2*log(2)), 0.5 + log(2), 0.5 + log(2), 3)
end

function default_runner_params()
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

function hypercube_constraints(; lower = -1, upper = 1)
    (X) -> clamp.(X, lower, upper)
end

function get_optimizer(runner_state::RunnerState)
    return runner_state.swarm.memory.global_best, runner_state.swarm.memory.global_best_score
end

end # module