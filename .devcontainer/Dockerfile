FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y \
    gcc-13 g++-13 \
    wget \
    curl \
    git \
    acl \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    build-essential \
    swig \
    python3-dev \
    cython3

# Ensure the latest version of libstdc++ is used
ENV CC=/usr/bin/gcc-13
ENV CXX=/usr/bin/g++-13
    
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

# Fix the libstdc++.so.6 issue for conda
RUN rm -f /home/scientist/miniconda3/lib/libstdc++.so.6

# Add Conda to the path
ENV PATH="/home/scientist/miniconda3/bin:$PATH"

RUN conda create -y -n meta python=3.11.2 && \
    conda clean -afy

RUN /home/scientist/miniconda3/bin/conda run -n meta conda install -y notebook

# Resolve libstdc++.so.6 issue for conda
RUN find /home/scientist/miniconda3/envs/meta/ -name "libstdc++.so.6" -delete || true

# Install additional python requirements
COPY requirements.txt .
RUN conda run -n meta pip install -r requirements.txt

# Install Julia
RUN curl -fsSL https://install.julialang.org | sh -s -- -y

# Add Julia to the path
ENV PATH="/home/scientist/.juliaup/bin:$PATH"

# Create default environment and start notebook
RUN mkdir -p /home/scientist/.julia/environments/default

# Copy the project file, install dependencies, and attach Jupyter notebooks for Julia
COPY Project.toml /home/scientist/.julia/environments/default/Project.toml
COPY Manifest.toml /home/scientist/.julia/environments/default/Manifest.toml
RUN julia -e 'using Pkg; Pkg.activate("/home/scientist/.julia/environments/default"); Pkg.instantiate(); Pkg.precompile(); using IJulia; IJulia.installkernel("Julia", "--project=/home/scientist/.julia/environments/default")'



# Expose the port
EXPOSE 8888

CMD ["bash"]