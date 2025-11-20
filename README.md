# Introduction
This polyglot repository includes metaheuristic optimization code written in Julia for solving constrained optimal design of experiments problems, and Python visualization code to analyze the results.


# Quick Start
## Julia
### Installation
```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.11.2
```

### Create Environment
Launch Julia from the command line:

```bash
julia +1.11.2 --project=.
```

Then run the following to start the IJulia kernel:

```julia
using Pkg
Pkg.instantiate()
Pkg.add("IJulia")

using IJulia
IJulia.installkernel(
    "MetaDoE";
    env = Dict(
        "JULIA_PROJECT"      => "@.",
        "JULIA_NUM_THREADS"  => "8",
    )
)
```

## Python
### Installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Environment
```bash
uv sync
uv run python -m ipykernel install --user --name meta --display-name "Python (meta)"
```
