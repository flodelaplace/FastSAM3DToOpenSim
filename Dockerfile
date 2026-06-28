# FastSAM3DToOpenSim — Docker image
# Base: CUDA 12.4 runtime (supports Ampere, Ada Lovelace, Hopper — G4dn/G5/G6/P4/P5)
# We use the `runtime` variant (not `devel`) because no source compilation is
# needed — PyTorch ships its own CUDA libs via pip, and detectron2 (the only
# thing that required nvcc at build-time) has been removed since the AWS
# pipeline uses YOLO for detection, not ViTDet/detectron2.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# --------------------------------------------------------------------------- #
# System packages                                                               #
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
# gcc + python3-dev are required at RUNTIME (not just build) because Triton
# (used by torch.compile) compiles CUDA kernel wrappers on the fly and needs a
# C compiler + Python.h available inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates unzip \
        ffmpeg \
        libgl1 libglib2.0-0 \
        libgomp1 libegl1 libxrender1 libxext6 \
        libsm6 libx11-6 \
        gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (needed for S3 pull/push in run_job.sh)
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip && \
    unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip

# s5cmd v2.2.2 (pinned) — sync S3 ~256 streams parallèles vs 10 pour
# `aws s3 sync`. Utilisé en chemin rapide pour télécharger les ~498 fichiers
# du backbone DINOv3 au cold start. run_job.sh garde un fallback automatique
# vers `aws s3 sync` si s5cmd échoue, donc cet install est purement additif.
RUN curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz" \
    | tar xz -C /usr/local/bin/ s5cmd && chmod +x /usr/local/bin/s5cmd && \
    /usr/local/bin/s5cmd version

# Node.js 20 + @gltf-transform/cli — pour compresser les GLB animés avec
# vrai Draco (extension KHR_draco_mesh_compression). gltfpack v1.1 ne fait
# QUE meshopt (EXT_meshopt_compression) qui n'est pas supporté nativement
# par Blender et cause des misalignments avec l'anatomical sur les meshes
# animés à cause de la quantization vertex lossy. Draco via gltf-transform
# préserve mieux l'alignement et est universellement reconnu (three.js,
# model-viewer, Babylon, Blender natif…).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g @gltf-transform/cli && \
    gltf-transform --version

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
# NB: `pip` ajouté explicitement dans conda create — les versions récentes de
# conda-forge python ne bundlent plus pip par défaut dans les nouveaux envs
# (changé courant 2026), il faut le demander explicitement sinon le pip
# install qui suit fait `No such file or directory: bin/pip`.
RUN conda create -y -n fast_sam_3d_body python=3.11 pip && \
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
# Pin TRT to 10.7.0 — la 11.1 a retiré NetworkDefinitionCreationFlag.EXPLICIT_BATCH
# et BuilderFlag.FP16, ce qui casse convert_backbone_tensorrt.py et
# convert_moge_encoder_trt.py (TRT 11+ requirerait une refonte des scripts).
RUN /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        "tensorrt-cu12==10.7.0" "tensorrt-cu12-bindings==10.7.0" "tensorrt-cu12-libs==10.7.0"

# Step 5: Application requirements (~5 min)
COPY docker/requirements_docker.txt /tmp/requirements_docker.txt
RUN /opt/conda/envs/fast_sam_3d_body/bin/pip install \
        -r /tmp/requirements_docker.txt

# Step 6: MoGe + utility git deps (~2 min)
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
