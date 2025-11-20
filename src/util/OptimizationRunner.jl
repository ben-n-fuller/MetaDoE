module OptimizationRunner
using ..PSO
using Base.Iterators
using Base.Threads

function run_pso_experiments(experiment, objectives, enforcers, runner_params, save_func)
    combos = collect(Iterators.product(objectives, enforcers))
    Threads.@threads for i in eachindex(combos)
        obj, enf = combos[i]

        println("Objective: $obj, Enforcer: $enf")

        context = PSO.create_context(
            experiment,
            obj;
            callback      = PSO.aggregate_results(; save_world = true),
            runner_params = runner_params,
            enforcer_type = enf,
        )

        runner_state, history = PSO.optimize(context)

        save_func(history, obj, enf)
        println("Best score: $(runner_state.swarm.memory.global_best_score)")
    end
end

function save_3d_results(; prefix = "")
    return (history, obj, enf) -> PSO.save_history_3d(history; location = "$(prefix)_$(obj)_$(enf).npy")
end

end # module