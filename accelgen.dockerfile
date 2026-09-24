# Use Ubuntu as base
FROM ubuntu:22.04

ARG http_proxy
ARG https_proxy

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3.11 \
    python3-pip \
    python3-venv \
    libedit-dev \
    libncurses5-dev \
    zlib1g-dev \
    libxml2-dev \
    lld \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a workspace
WORKDIR /opt
RUN git clone --recursive https://github.com/llvm/circt.git \
    && cd ./circt \
    && git checkout d3e792bc8717a9947d10505a02a6bf8a511e4a43 \
    && git submodule update --init --recursive \
    && pip install psutil \
    && pip install torch==2.8.0 \
    && pip install torchvision==0.23.0 \
    && pip install torch-mlir==20250607.491 -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels


WORKDIR /opt/circt
RUN cmake -G Ninja llvm/llvm -B build \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="circt" \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=/usr/local/circt \
    -DMLIR_INCLUDE_DOCS=ON \
    && ninja -C build \
    && ninja -C build check-mlir check-circt \
    && ninja -C build install

WORKDIR /opt  
RUN git config --global url."https://github.com/".insteadOf git@github.com: \
    && git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git \
    && cd accelergy-timeloop-infrastructure \
    && pip3 install pyyaml \
    && mkdir -p /root/.local/bin \
    && sed -i 's/sudo //g' Makefile \
    && make install_accelergy \
    && pip3 install ./src/timeloopfe \
    && make install_timeloop



WORKDIR /opt  
RUN git clone https://github.com/HewlettPackard/cacti.git \
    && cd cacti \
    && make 

# Set environment variables
ENV PATH="/usr/local/circt/bin:/usr/local/bin:/opt/cacti:${PATH}"

# Default shell
CMD ["/bin/bash"]