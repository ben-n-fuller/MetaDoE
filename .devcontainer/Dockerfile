# syntax=docker/dockerfile:1

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    build-essential pkg-config \
    libssl-dev libxml2-dev libcurl4-openssl-dev libgit2-dev \
    sudo acl tar bzip2 jq \
 && rm -rf /var/lib/apt/lists/*

ARG USERNAME=scientist
ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd  --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USERNAME} \
 && chmod 0440 /etc/sudoers.d/90-${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV CONDA_DIR=/home/${USERNAME}/miniconda3
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Pick correct Miniforge installer by arch
RUN set -euo pipefail; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64)  mf_arch="x86_64" ;; \
      arm64)  mf_arch="aarch64" ;; \
      *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${mf_arch}.sh"; \
    curl -fsSL "$url" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh && \
    conda config --system --remove-key channels || true && \
    conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict && \
    conda install -y python=3.11 jupyterlab ipykernel && \
    conda clean -afy

ARG JULIA_VERSION=1.11.2
ENV JULIA_DEPOT_PATH=/home/${USERNAME}/.julia

RUN set -euo pipefail; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64)  jl_dir_arch="x64";    jl_file_arch="x86_64" ;; \
      arm64)  jl_dir_arch="aarch64"; jl_file_arch="aarch64" ;; \
      *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    minor="${JULIA_VERSION%.*}"; \
    url="https://julialang-s3.julialang.org/bin/linux/${jl_dir_arch}/${minor}/julia-${JULIA_VERSION}-linux-${jl_file_arch}.tar.gz"; \
    echo "Downloading Julia from: ${url}"; \
    curl -fsSL "$url" -o /tmp/julia.tar.gz; \
    sudo mkdir -p /opt/julia; \
    sudo tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1; \
    rm -f /tmp/julia.tar.gz; \
    sudo ln -sf /opt/julia/bin/julia /usr/local/bin/julia


RUN echo 'eval "$('"'"'${CONDA_DIR}/bin/conda'"'"' shell.bash hook)"' >> ~/.bashrc \
 && echo 'export PATH=${CONDA_DIR}/bin:$PATH' >> ~/.bashrc

ENV TERM=xterm-256color
EXPOSE 8888

WORKDIR /workspaces
