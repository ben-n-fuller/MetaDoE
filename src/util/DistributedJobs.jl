module DistributedJobs

export Job, JobResult
export run_jobs, run_jobs_no_save, load_and_concatenate, load

using HDF5
using Dates
using Logging
using Distributed

struct Job
    name::String
    job_handler::Function
    data_generator::Function
end

struct ComputeResult
    data::Any
    job::Job
end

struct JobResult
    name::String
    file_name::String
end

function create_job(handler::Function, generator::Function; name = "compute_job")
    Job(name, handler, generator)
end

function create_compute_result(result::Array, job::Job)
    ComputeResult(result, job)
end

function apply_handler_to_batch(job::Job)
    transformed_data = (job.job_handler ∘ job.data_generator)()
    return create_compute_result(transformed_data, job)
end

function save_job_result(compute_result::ComputeResult, path_prefix::String)
    ts_str = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    file_path = joinpath(path_prefix, "$(compute_result.job.name) $ts_str.h5")
    @info "Saving job '$(compute_result.job.name)' to '$file_path'"
    h5write(file_path, compute_result.job.name, compute_result.data)
    return JobResult(compute_result.job.name, file_path)
end

function run_jobs(jobs::Vector{Job}; save_function::Function = save_job_result, path_prefix = ".")::Vector{JobResult}
    @info "Distributing $(length(jobs)) jobs..."
    save_function_with_path = (r) -> save_function(r, path_prefix)
    pmap(save_function_with_path ∘ apply_handler_to_batch, jobs)
end

function run_jobs_no_save(jobs::Vector{Job})::Vector{ComputeResult}
    @info "Distributing $(length(jobs)) jobs..."
    pmap(apply_handler_to_batch, jobs)
end

function load(completed_jobs::Vector{JobResult})
    map((res) -> h5read(res.file_name, res.name), completed_jobs)
end

function load_and_concatenate(completed_jobs::Vector{JobResult})
    all_data = load(completed_jobs)
    return cat(all_data...; dims=1)
end

end
