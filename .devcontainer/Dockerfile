FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install dependencies for Jupyter Lab
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    acl \
    python3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev
    
# Create the user
RUN groupadd --gid 1001 scientist \
    && useradd --uid 1001 --gid 1001 -m scientist \
    && apt-get update \
    && apt-get install -y sudo \
    && echo scientist ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/scientist \
    && chmod 0440 /etc/sudoers.d/scientist \
    && chown -R scientist:scientist /usr/src/app /usr/local
    
# Set the user
USER scientist

# Detect architecture and install Miniconda
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    bash ~/miniconda.sh -b -p ~/miniconda3 && \
    rm ~/miniconda.sh

# Add Conda to the path
ENV PATH="/home/scientist/miniconda3/bin:$PATH"

RUN conda install -y notebook

# Install Julia
RUN curl -fsSL https://install.julialang.org | sh -s -- -y

# Add Julia to the path
ENV PATH="/home/scientist/.juliaup/bin:$PATH"

# Create default environment and start notebook
RUN mkdir -p /home/scientist/.julia/environments/default

# Copy the project file, install dependencies, and attach Jupyter notebooks for Julia
COPY Project.toml /home/scientist/.julia/environments/default/Project.toml
COPY Manifest.toml /home/scientist/.julia/environments/default/Manifest.toml
RUN julia -e 'using Pkg; Pkg.activate("/home/scientist/.julia/environments/default"); Pkg.instantiate(); Pkg.precompile(); using IJulia; notebook(detached=true)'

# Install additional python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8888

CMD ["bash"]