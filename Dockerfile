# FastSAM3DToOpenSim — Docker image
# Base: CUDA 12.4 (supports Ampere, Ada Lovelace, Hopper — G4dn/G5/G6/P4/P5)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# --------------------------------------------------------------------------- #
# System packages                                                               #
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates unzip \
        ffmpeg \
        libgl1 libglib2.0-0 \
        libgomp1 libegl1 libxrender1 libxext6 \
        libsm6 libx11-6 \
        blender \
        build-essential cmake python3-dev libffi-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (needed for S3 pull/push in run_job.sh)
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip && \
    unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip

# --------------------------------------------------------------------------- #
# Miniforge → /opt/conda                                                        #
# --------------------------------------------------------------------------- #
RUN wget -q \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# --------------------------------------------------------------------------- #
# Env 1: fast_sam_3d_body (Python 3.11)                                         #
# Split into multiple RUN layers so a transient network failure doesn't        #
# wipe out hours of work, and Docker can cache each step independently.        #
# --------------------------------------------------------------------------- #

# Configure pip with retries to survive flaky network
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    PIP_NO_CACHE_DIR=1

# Step 1: Create env + base build tools (~30s)
RUN conda create -y -n fast_sam_3d_body python=3.11 && \
    /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        numpy cython setuptools virtualenv

# Step 2: chumpy from conda-forge (cached separately) (~1 min)
RUN conda install -y -n fast_sam_3d_body -c conda-forge chumpy && \
    conda clean -afy

# Step 3: PyTorch (~3-5 min, the big download ~3 GB)
RUN /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        --extra-index-url https://download.pytorch.org/whl/cu124

# Step 4: TensorRT (~2 min)
# Engines are GPU-specific AND TRT-version-specific. They are always regenerated
# on first launch via docker/entrypoint.sh (GENERATE_TRT=1 by default).
RUN /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs

# Step 5: Application requirements (~5 min)
COPY docker/requirements_docker.txt /tmp/requirements_docker.txt
RUN /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        -r /tmp/requirements_docker.txt

# Step 6: detectron2 (compiles from source, ~5-8 min)
RUN CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES="" \
    /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        --no-build-isolation --no-deps \
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"

# Step 7: MoGe + utility git deps (~2 min)
RUN /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        "git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8" \
        "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6" \
        "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"

# Final cleanup
RUN conda clean -afy && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------- #
# Env 2: opensim (Python 3.10, opensim-org channel)                            #
# --------------------------------------------------------------------------- #
RUN conda create -y -n opensim -c opensim-org -c conda-forge python=3.10 opensim && \
    conda clean -afy

# Set an environment variable to the exact path of the opensim python interpreter
# This makes it easy for the main script to find it without relying on `conda run`
ENV OPENSIM_PYTHON_PATH /opt/conda/envs/opensim/bin/python

# --------------------------------------------------------------------------- #
# Application code                                                              #
# --------------------------------------------------------------------------- #
WORKDIR /app
COPY . /app

# Ensure output dir exists
RUN mkdir -p /outputs

# Make scripts executable (already copied by COPY . /app)
RUN chmod +x /app/docker/entrypoint.sh /app/scripts/run_job.sh && \
    ln -sf /app/docker/entrypoint.sh /entrypoint.sh

# Volumes for persistent data
VOLUME ["/app/checkpoints", "/app/videos", "/outputs"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
