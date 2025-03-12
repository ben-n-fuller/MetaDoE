# Introduction
This repository contains Julia code for computational research in industrial statistics.

- `/docs` contains relevant documentation for the team including code usage and other instructions
- `/src` contains a library of modules defining common operations
- `/notebooks` contains exploratory Jupyter notebooks for developing code and ideas
- `/examples` contains examples of code use and common workflows

# Usage
## Dev Containers (Recommended)
A dev container is a portable environment that provides necessary dependencies for software development. The dev container in this project provides Jupyter notebook support for Python, Julia, and R, as well as many packages for each language. 

### Auth
You will need to generate a Personal Access Token (PAT) from GitHub to gain access to the container. See [Working with the container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) for more details.
1. In GitHub, click your profile image at the top right and select `Settings`
2. Open `Developer Settings` at the bottom
3. Select `Personal access tokens->Tokens (classic)`
4. Click to generate a new classic token
5. Add a note, change the expiration, and select the `write:packages` option
6. Click `Generate token`
7. Run the following (in Mac or WSL):
```
export CR_PAT=YOUR_TOKEN
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

You should see a `Login Succeeded` message.

### Run
0. **(Windows Only)** Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
1. Install the [Docker Engine](https://docs.docker.com/engine/install/)
2. Open the project root in VS Code
3. Press `Ctrl+Shift+P` and run `Dev Containers: Open Folder in Container`
4. Select the root folder `Pearson-Exploratorium`

The first time the dev container is created, it may take a long time to load as the image is downloaded and the dependencies installed.

### Using Jupyter Notebooks
1. Create a new notebook with `Ctrl+Shift+P` and type `Create: New Jupyter notebook`
2. In the top right, choose from one of the following kernels (more may be available through `Select another kernel->Jupyter Kernel` or `Select another kernel->Python environments`)
   
| Kernel | Description |
|-|-|
| `Julia 1.x.x` | Julia kernel. **Don't** use the release channel |
| `base (Python 3.x.x)` | Python kernel provided by conda. **Don't** use any other python kernel |
| `R` | R kernel |

## Local Julia Installation
If Julia is not already installed, `juliaup` can be very useful for installing and managing Julia versions. Instructions for different platforms are available in the [official repo](https://github.com/JuliaLang/juliaup).

Once installed, activate the local package environment in the root directory:

```
julia -e "import Pkg; Pkg.activate(\".\"); Pkg.instantiate()"
```

### IDE Configuration
1. Install the official `Jupyter` extension
2. Install the official `Julia` language extension
3. Open the root folder in vscode, open a notebook, and select the `Release Channel` kernel in the top right
