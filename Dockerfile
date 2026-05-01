# ---- Stage 1: Build C++ environment core ----
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git python3 python3-pip \
    libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install pybind11.
RUN git clone --depth 1 --branch v2.11.1 https://github.com/pybind/pybind11.git && \
    cd pybind11 && cmake -S . -B build && cmake --install build

# Build C++ environment bindings.
COPY env/core ./env/core
COPY env/bindings ./env/bindings
COPY CMakeLists.txt .
RUN cd env/core && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# ---- Stage 2: Python training environment ----
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    libopenblas0 libomp5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts from builder.
COPY --from=builder /build/env/core/build ./env/core/build

# Copy Python source.
COPY pyproject.toml .
COPY env/__init__.py ./env/
COPY env/tetris_env.py ./env/
COPY env/state_encoder.py ./env/
COPY env/reward_calculator.py ./env/
COPY agent/ ./agent/
COPY trainer/ ./trainer/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY configs/ ./configs/

# Install Python dependencies.
RUN pip install --no-cache-dir -e ".[train,inference]"

# Default command: start training.
CMD ["python3", "scripts/train.py"]
