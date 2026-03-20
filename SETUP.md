# Setup Guide

This guide covers setting up the FastSAM3DToOpenSim pipeline on a new machine.

---

## Prerequisites

- Linux (tested on Ubuntu 22.04 / 24.04)
- NVIDIA GPU — see GPU compatibility section below
- NVIDIA driver 570+ (for CUDA 12.8 support)
- Conda (Miniconda or Anaconda)
- ~30 GB free disk space (checkpoints + TRT engines)

---

## Step 1: Clone the repository

```bash
git clone https://github.com/AitorIriondo/FastSAM3DToOpenSim.git
cd FastSAM3DToOpenSim
git checkout fix   # the working branch
```

---

## Step 2: Create the conda environment

```bash
conda create -n fast_sam_3d_body python=3.11 -y
conda activate fast_sam_3d_body
```

---

## Step 3: Install PyTorch

**IMPORTANT: the correct PyTorch version depends on your GPU architecture. See the GPU section below.**

For Blackwell (RTX 5000 series) — CUDA 12.8:
```bash
pip install torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

For Ada Lovelace / Ampere (RTX 3000/4000 series, A-series) — CUDA 12.4:
```bash
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

---

## Step 4: Install Python dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Additional packages needed
pip install onnx onnxscript onnxruntime-gpu
pip install pygltflib==1.16.5
pip install trimesh

# Do NOT use `conda install` inside this env — it is broken for this env
# (conda's solver pulls in incompatible packages). Use pip for everything.
```

---

## Step 5: Download checkpoints

### SAM-3D-Body model (from Meta / Hugging Face)

```bash
mkdir -p checkpoints/sam-3d-body-dinov3
# Download from Hugging Face:
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='facebook/sam-3d-body',
    local_dir='./checkpoints/sam-3d-body-dinov3',
)
"
```

Required files after download:
```
checkpoints/sam-3d-body-dinov3/
  model.ckpt           (~2 GB)
  assets/
    mhr_model.pt       (~664 MB)
    [other config files]
```

### YOLO pose model

```bash
mkdir -p checkpoints/yolo
# Download yolo11m-pose.pt from Ultralytics:
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
mv yolo11m-pose.pt checkpoints/yolo/
```

---

## Step 6: Build TensorRT engines

TRT engines are **GPU-specific** — they must be built on the machine where they will be used. They cannot be copied from another machine.

### 6a. YOLO TRT engine

```bash
python -c "
from ultralytics import YOLO
model = YOLO('checkpoints/yolo/yolo11m-pose.pt')
model.export(format='engine', device=0, half=True, imgsz=640)
"
mv checkpoints/yolo/yolo11m-pose.engine checkpoints/yolo/  # already there if exported in place
```

### 6b. MoGe TRT encoder

```bash
python convert_moge_encoder_trt.py
```

**Known issue**: the script must use `parse_from_file()` instead of `parse(f.read())` for loading the ONNX model with external weights. If it fails with an error about missing weights, edit the script and change:
```python
# Wrong:
parser.parse(f.read(), ...)
# Correct:
parser.parse_from_file(onnx_path)
```

Output: `checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine` (~46 MB)

### 6c. SAM-3D-Body backbone TRT engine

```bash
python convert_backbone_tensorrt.py
```

**Known issue**: the script needs `dynamo=False` in the `torch.onnx.export` call, otherwise it fails on newer PyTorch. Edit the script and add `dynamo=False`:
```python
torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, ...)
```

This builds for fixed 512×512 input with dynamic batch [1, 2, 4].
Takes 10–30 minutes depending on GPU.

Output: `checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine` (~1.6 GB)

---

## GPU Compatibility

### GPU Architecture Reference

| GPU | Architecture | sm_ | CUDA min | TRT min |
|-----|-------------|-----|----------|---------|
| RTX 5090 (Desktop) | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 5090 Laptop | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 5080 (Desktop) | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 5070 Ti (Desktop) | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 5070 Ti Laptop | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 5070 (Desktop) | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 5070 Laptop | Blackwell | sm_120 | 12.8 | 10.7 |
| RTX 4090 (Desktop) | Ada Lovelace | sm_89 | 11.8 | 8.6 |
| RTX 4080 / 4070 | Ada Lovelace | sm_89 | 11.8 | 8.6 |
| RTX 3090 / 3080 | Ampere | sm_86 | 11.1 | 8.0 |
| RTX A5000 / A4000 | Ampere | sm_86 | 11.1 | 8.0 |
| RTX A3000 Laptop | Ampere | sm_86 | 11.1 | 8.0 |
| RTX A3000 Desktop (non-existent — A-series laptops only) | — | — | — | — |

### Blackwell (RTX 5000 series) — Desktop and Laptop

Blackwell is the newest architecture (2025). CUDA 12.8 and TRT 10.7 or newer are required.

**PyTorch**: as of mid-2025, you need PyTorch 2.6+ or the cu128 nightly builds. The tested version in this repo is `torch==2.10.0+cu128`.

```bash
pip install torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

**TensorRT**: install TensorRT 10.7+ via pip:
```bash
pip install tensorrt==10.7.0 tensorrt-cu12==10.7.0
```

**Expected performance** (approx, varies by thermal/power limits):
| GPU | Expected FPS (body only, 512px) |
|-----|-------------------------------|
| RTX 5090 Desktop | ~18-22 fps |
| RTX 5090 Laptop | ~14-16 fps |
| RTX 5080 Desktop | ~15-18 fps |
| RTX 5070 Ti Desktop | ~12-15 fps |
| RTX 5070 Ti Laptop | ~10-13 fps |
| RTX 5070 Desktop | ~10-13 fps |
| RTX 5070 Laptop | ~8-11 fps |

Laptop GPUs run slower than their desktop counterparts due to lower TDP (power limits). Under thermal throttling, sustained FPS can be 20-30% lower than peak.

### Ada Lovelace (RTX 4000 series)

Well-supported by PyTorch 2.x and TRT 8.6+.

```bash
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
pip install tensorrt==10.3.0 tensorrt-cu12==10.3.0
```

**Expected performance**:
| GPU | Expected FPS (body only, 512px) |
|-----|-------------------------------|
| RTX 4090 Desktop | ~16-20 fps |
| RTX 4090 Laptop | ~10-14 fps |
| RTX 4080 Desktop | ~12-16 fps |
| RTX 4070 Ti | ~9-12 fps |

### Ampere (RTX 3000 series, A-series Workstation)

Standard CUDA 12.x support.

```bash
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
pip install tensorrt==10.3.0 tensorrt-cu12==10.3.0
```

**Note for RTX A3000 Laptop** (96W / 80W variants): the A3000 is a mid-range workstation mobile GPU with 6 GB VRAM. The backbone TRT engine (~1.6 GB) may be tight on VRAM together with the rest of the model. If you hit OOM, reduce backbone batch size when building the engine or fall back to `USE_TRT_BACKBONE=0`.

**Expected performance**:
| GPU | Expected FPS (body only, 512px) |
|-----|-------------------------------|
| RTX 3090 Desktop | ~9-12 fps |
| RTX 3080 Desktop | ~8-10 fps |
| RTX A5000 | ~9-12 fps |
| RTX A4000 | ~7-9 fps |
| RTX A3000 Laptop | ~4-6 fps |

---

## Step 7: (Optional) SMPL data files for ZMQ streaming

The `run_publisher.py` ZMQ real-time streaming script requires additional data files that are not part of this repo:

1. `mhr2smpl/data/SMPL_NEUTRAL.pkl` — from https://smpl-x.is.tue.mpg.de/ (free academic registration required)
2. `mhr2smpl/data/mhr2smpl_mapping.npz` — from the MHR repository at `tools/mhr_smpl_conversion/assets/`

Without these files, `demo_video_opensim.py` still works fully (TRC, MOT, GLB). Only the real-time ZMQ publisher is blocked.

---

## Known issues and workarounds

| Issue | Workaround |
|-------|-----------|
| `conda install` breaks the env | Use `pip install` for everything in `fast_sam_3d_body` |
| `nvcc` not installed (no sudo) | detectron2 compiles CPU-only; not used in inference, so harmless |
| `convert_moge_encoder_trt.py` fails on external weights | Change `parse(f.read())` to `parse_from_file(onnx_path)` |
| `convert_backbone_tensorrt.py` export fails | Add `dynamo=False` to `torch.onnx.export()` |
| TRT engine not found at startup | Build the engine on the target machine; do not copy from another machine |
| OOM on A3000 or low-VRAM GPU | Set `USE_TRT_BACKBONE=0`; use PyTorch backbone instead |
| First run is slow (30-60s) | Normal — `torch.compile` is doing JIT compilation. Subsequent runs are fast. |
| TRT engine size 1.6 GB won't load | Ensure you have enough VRAM + system RAM. Kill other GPU processes first. |
