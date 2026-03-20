# Windows Setup Guide

This guide covers setting up the FastSAM3DToOpenSim pipeline on Windows 10/11.

---

## Important Windows-specific limitations

Before starting, be aware of two differences vs Linux:

1. **`torch.compile` is experimental on Windows.** It requires a Windows build of Triton which is not part of the standard pip package as of 2025. You can try installing `triton-windows` (community port), but if it fails, set `USE_COMPILE=0` and `DECODER_COMPILE=0`. Expect ~2-3 fps lower than the Linux numbers as a result.

2. **`detectron2` has no official Windows wheel.** It is not used during inference, so this is harmless — skip it.

---

## Prerequisites

Install all of the following before starting:

### 1. NVIDIA driver
Download and install the latest Game Ready or Studio driver from https://www.nvidia.com/drivers

Minimum driver version: **572.x** (for CUDA 12.8 / Blackwell)

### 2. CUDA Toolkit
On Windows, CUDA is NOT included with the driver alone — you must install the toolkit separately.

Download CUDA Toolkit 12.8 from https://developer.nvidia.com/cuda-downloads
Select: Windows → x86_64 → your Windows version → exe (local)

After installing, verify in a new Command Prompt:
```cmd
nvcc --version
```

### 3. Visual Studio Build Tools 2022
Required for compiling C extensions (some Python packages need it).

Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
In the installer, select **"Desktop development with C++"**.

### 4. Miniconda
Download from https://docs.conda.io/en/latest/miniconda.html
Use the Windows 64-bit installer. During install, tick **"Add Miniconda to PATH"** (or use Anaconda Prompt for all commands below).

### 5. Git
Download from https://git-scm.com/download/win

---

## Step 1: Clone the repository

Open **Anaconda Prompt** (or any terminal with conda available):

```cmd
git clone https://github.com/AitorIriondo/FastSAM3DToOpenSim.git
cd FastSAM3DToOpenSim
git checkout fix
```

---

## Step 2: Create the conda environment

```cmd
conda create -n fast_sam_3d_body python=3.11 -y
conda activate fast_sam_3d_body
```

---

## Step 3: Install PyTorch

**For Blackwell (RTX 5000 series) — CUDA 12.8:**
```cmd
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For Ada Lovelace / Ampere (RTX 3000/4000 series) — CUDA 12.4:**
```cmd
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

---

## Step 4: Install Python dependencies

```cmd
pip install -r requirements.txt
pip install onnx onnxscript onnxruntime-gpu
pip install pygltflib==1.16.5
pip install trimesh
pip install tensorrt==10.7.0 tensorrt-cu12==10.7.0
```

**Do NOT use `conda install` inside this env — use pip for everything.**

### Optional: torch.compile on Windows

If you want to try `torch.compile` (experimental):
```cmd
pip install triton-windows
```

If installation fails or inference crashes with a Triton error, disable it:
```cmd
set USE_COMPILE=0
set DECODER_COMPILE=0
```

---

## Step 5: Download checkpoints

```cmd
pip install huggingface_hub ultralytics
```

### SAM-3D-Body model
```cmd
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/sam-3d-body', local_dir='./checkpoints/sam-3d-body-dinov3')"
```

### YOLO pose model
```cmd
python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
move yolo11m-pose.pt checkpoints\yolo\
```

---

## Step 6: Build TensorRT engines

TRT engines must be built on the target machine. They are architecture-specific and cannot be copied from Linux.

### 6a. YOLO TRT engine
```cmd
python -c "from ultralytics import YOLO; m = YOLO('checkpoints/yolo/yolo11m-pose.pt'); m.export(format='engine', device=0, half=True, imgsz=640)"
```

### 6b. MoGe TRT encoder
```cmd
python convert_moge_encoder_trt.py
```
If it fails with a weights error, see the known issue in SETUP.md (change `parse(f.read())` to `parse_from_file(onnx_path)`).

### 6c. SAM-3D-Body backbone TRT engine
```cmd
python convert_backbone_tensorrt.py
```
If it fails, see the known issue in SETUP.md (add `dynamo=False` to `torch.onnx.export`).

This takes 10–30 minutes.

---

## Running the pipeline on Windows

On Windows, environment variables are set differently. Use either **Command Prompt** or **PowerShell** — the syntax differs.

### Command Prompt (cmd.exe)

```cmd
conda activate fast_sam_3d_body

set SKIP_KEYPOINT_PROMPT=1
set FOV_TRT=1
set FOV_FAST=1
set FOV_MODEL=s
set FOV_LEVEL=0
set USE_TRT_BACKBONE=1
set USE_COMPILE=1
set DECODER_COMPILE=1
set COMPILE_MODE=reduce-overhead
set MHR_NO_CORRECTIVES=1
set GPU_HAND_PREP=1
set BODY_INTERM_PRED_LAYERS=0,2
set DEBUG_NAN=0
set PARALLEL_DECODERS=0
set COMPILE_WARMUP_BATCH_SIZES=1

python demo_video_opensim.py ^
    --video_path videos\aitor_garden_walk.mp4 ^
    --detector_model checkpoints\yolo\yolo11m-pose.engine ^
    --inference_type body ^
    --fx 1371
```

### PowerShell

```powershell
conda activate fast_sam_3d_body

$env:SKIP_KEYPOINT_PROMPT = "1"
$env:FOV_TRT = "1"
$env:FOV_FAST = "1"
$env:FOV_MODEL = "s"
$env:FOV_LEVEL = "0"
$env:USE_TRT_BACKBONE = "1"
$env:USE_COMPILE = "1"
$env:DECODER_COMPILE = "1"
$env:COMPILE_MODE = "reduce-overhead"
$env:MHR_NO_CORRECTIVES = "1"
$env:GPU_HAND_PREP = "1"
$env:BODY_INTERM_PRED_LAYERS = "0,2"
$env:DEBUG_NAN = "0"
$env:PARALLEL_DECODERS = "0"
$env:COMPILE_WARMUP_BATCH_SIZES = "1"

python demo_video_opensim.py `
    --video_path videos\aitor_garden_walk.mp4 `
    --detector_model checkpoints\yolo\yolo11m-pose.engine `
    --inference_type body `
    --fx 1371
```

### Convenience batch file

Save this as `run_opensim.bat` in the repo root and edit the video path:

```bat
@echo off
conda activate fast_sam_3d_body

set SKIP_KEYPOINT_PROMPT=1
set FOV_TRT=1
set FOV_FAST=1
set FOV_MODEL=s
set FOV_LEVEL=0
set USE_TRT_BACKBONE=1
set USE_COMPILE=1
set DECODER_COMPILE=1
set COMPILE_MODE=reduce-overhead
set MHR_NO_CORRECTIVES=1
set GPU_HAND_PREP=1
set BODY_INTERM_PRED_LAYERS=0,2
set DEBUG_NAN=0
set PARALLEL_DECODERS=0
set COMPILE_WARMUP_BATCH_SIZES=1

python demo_video_opensim.py ^
    --video_path %1 ^
    --detector_model checkpoints\yolo\yolo11m-pose.engine ^
    --inference_type body ^
    --fx %2

pause
```

Usage:
```cmd
run_opensim.bat videos\my_video.mp4 1371
```

---

## Expected performance on Windows

Windows adds some overhead vs Linux due to driver scheduling and WDDM (Windows Display Driver Model). WDDM batches GPU commands differently than Linux's TCC mode. Expect roughly **10-20% lower FPS** than the Linux numbers for the same GPU.

| GPU | Linux FPS (body) | Windows FPS (body, estimated) |
|-----|-----------------|-------------------------------|
| RTX 5090 Desktop | ~18-22 | ~15-18 |
| RTX 5090 Laptop | ~14-16 | ~11-14 |
| RTX 5080 Desktop | ~15-18 | ~12-15 |
| RTX 5070 Ti Desktop | ~12-15 | ~10-13 |
| RTX 5070 Ti Laptop | ~10-13 | ~8-11 |
| RTX 5070 Desktop | ~10-13 | ~8-11 |
| RTX 5070 Laptop | ~8-11 | ~6-9 |
| RTX 4090 Desktop | ~16-20 | ~13-17 |
| RTX 4090 Laptop | ~10-14 | ~8-12 |
| RTX 3090 Desktop | ~9-12 | ~7-10 |
| RTX A3000 Laptop | ~4-6 | ~3-5 |

If `torch.compile` (triton-windows) is not working, subtract another ~2-3 fps from the above.

---

## Windows-specific known issues

| Issue | Workaround |
|-------|-----------|
| `triton` not found / Triton crash | Set `USE_COMPILE=0 DECODER_COMPILE=0`. Loses ~2-3 fps. |
| `RuntimeError: CUDA error: no kernel image available` | Your TRT engine was built for a different GPU arch. Rebuild on this machine. |
| Long path errors during pip install | Enable long paths: run `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f` as Administrator |
| `cv2.VideoWriter` produces empty MP4 | Install `opencv-python` not `opencv-python-headless`; also ensure the video codec is available: `pip install opencv-python` |
| Conda not recognised in PowerShell | Run `conda init powershell` once, then restart PowerShell |
| `torch.compile` warning about `reduce-overhead` | On Windows without triton, change `COMPILE_MODE=default` |
| TRT engine build fails with `OSError` on long paths | Move the repo to a shorter path, e.g. `C:\fsab\` |
