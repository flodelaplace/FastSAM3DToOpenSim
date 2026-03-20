# Pipeline Settings

All settings used in the current best-performance configuration.

---

## Full run command

```bash
SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 \
USE_TRT_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 COMPILE_MODE=reduce-overhead \
MHR_NO_CORRECTIVES=1 GPU_HAND_PREP=1 BODY_INTERM_PRED_LAYERS=0,2 \
DEBUG_NAN=0 PARALLEL_DECODERS=0 COMPILE_WARMUP_BATCH_SIZES=1 \
python demo_video_opensim.py \
    --video_path ./videos/aitor_garden_walk.mp4 \
    --detector yolo_pose \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --inference_type body \
    --fx 1371
```

---

## Environment variables — what each one does

### Backbone

| Variable | Value | What it does |
|----------|-------|-------------|
| `USE_TRT_BACKBONE` | `1` | Load the DINOv3 backbone from a pre-built TensorRT engine instead of running it in PyTorch. Saves ~40-50ms/frame on the backbone alone. The engine file must exist at `checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine`. |
| `USE_COMPILE` | `1` | Apply `torch.compile` to the backbone (active when `USE_TRT_BACKBONE=0`). Not used when TRT backbone is active, but harmless to leave on. |

### Decoder

| Variable | Value | What it does |
|----------|-------|-------------|
| `DECODER_COMPILE` | `1` | Apply `torch.compile` to the body and hand decoders. Reduces decoder time from ~80ms to ~54ms on RTX 5090 Laptop. |
| `COMPILE_MODE` | `reduce-overhead` | The `torch.compile` mode. `reduce-overhead` minimises kernel launch overhead (best for small batches). Other options: `default` (safer), `max-autotune` (longer compile, marginally faster). |
| `COMPILE_WARMUP_BATCH_SIZES` | `1` | Comma-separated list of batch sizes to warm up during the compilation step. Setting this to `1` avoids compiling for batch sizes that never occur in practice, cutting warmup time. |
| `BODY_INTERM_PRED_LAYERS` | `0,2` | Which intermediate layers of the body decoder to run prediction heads on. The decoder has 3 layers (0, 1, 2). Using `0,2` skips layer 1, saving ~15ms/frame with minor accuracy loss. Using `0,1,2` is full accuracy. |
| `PARALLEL_DECODERS` | `0` | Whether to run body and hand decoders in parallel CUDA streams. Tested as slightly slower (4.84fps vs 5.5fps before reaching 14fps) due to CUDA graph conflicts. Keep at 0. |

### Person detection (YOLO)

| Variable | Value | What it does |
|----------|-------|-------------|
| (CLI) `--detector` | `yolo_pose` | Uses YOLOv11 pose model for person detection and keypoint initialisation. |
| (CLI) `--detector_model` | `yolo11m-pose.engine` | The TRT-compiled YOLO engine. Using the engine instead of the `.pt` file saves ~20ms/frame. |

### Depth / FOV estimation (MoGe)

MoGe estimates the camera field-of-view when `--fx` is not provided. When `--fx` is provided, it still runs for depth conditioning but uses less compute.

| Variable | Value | What it does |
|----------|-------|-------------|
| `FOV_TRT` | `1` | Run the MoGe depth encoder via TRT engine. Saves a few ms/frame. Requires `checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine`. |
| `FOV_FAST` | `1` | Enable fast mode: skips some MoGe post-processing passes. |
| `FOV_MODEL` | `s` | Use the small MoGe model variant. Options: `s` (fast), `b` (balanced), `l` (accurate). |
| `FOV_LEVEL` | `0` | MoGe resolution level. `0` = coarsest/fastest, higher = finer depth map. Relevant mainly when depth conditioning is important; for body pose at 4-6 m, level 0 is sufficient. |

### Mesh / corrective shapes

| Variable | Value | What it does |
|----------|-------|-------------|
| `MHR_NO_CORRECTIVES` | `1` | Skip corrective blend shapes in the MHR mesh. These are small per-joint deformations that improve mesh quality at extreme poses (e.g., elbow crease). Skipping saves ~3-5ms/frame. Skeleton joint positions are not affected. |
| `GPU_HAND_PREP` | `1` | Perform hand crop preprocessing on GPU instead of CPU. Saves a few ms when hands are present. With `--inference_type body` this has no effect since the hand decoder is skipped entirely. |

### Keypoint prompt

| Variable | Value | What it does |
|----------|-------|-------------|
| `SKIP_KEYPOINT_PROMPT` | `1` | Skip feeding the 2D YOLO keypoints as a conditioning prompt into the decoder. This saves the cost of constructing the prompt token but may reduce robustness in difficult poses or partial occlusions. |

### Debug / safety

| Variable | Value | What it does |
|----------|-------|-------------|
| `DEBUG_NAN` | `0` | Disable NaN checking in intermediate tensors. NaN checks add CPU synchronisation points that break async GPU execution. Always 0 in production. |

---

## CLI flags — what each one does

| Flag | Value used | What it does |
|------|-----------|-------------|
| `--inference_type` | `body` | Skip the hand decoder entirely. No wrist-distal or finger joints are estimated. This is the single largest speed gain (~30% faster than `full`). Use `full` only if you need hand/finger motion data. |
| `--fx` | `1371` | Camera focal length in pixels. Providing this skips the MoGe-based FOV estimation and feeds a pinhole model directly. Accurate focal length improves the 3D position scale. For the test video shot on a specific phone, `1371` was calibrated. |
| `--detector_model` | `.engine` | Using the TensorRT YOLO engine instead of the `.pt` PyTorch model. |

---

## TensorRT engine specifications

### YOLO pose engine
- File: `checkpoints/yolo/yolo11m-pose.engine`
- Size: ~42.6 MB
- Input: 640×640, FP16
- Batch: dynamic (built by Ultralytics export)
- Architecture: sm_120 (Blackwell) — must rebuild on other GPUs

### MoGe encoder engine
- File: `checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine`
- Size: ~46.4 MB
- Input: variable (uses dynamic shapes)
- Precision: FP16
- Architecture: sm_120 — must rebuild on other GPUs

### DINOv3 backbone engine
- File: `checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine`
- Size: ~1.6 GB
- Input: fixed 512×512, FP16
- Batch: dynamic [1, 2, 4]
- Architecture: sm_120 — must rebuild on other GPUs
- Build time: 10–30 minutes

---

## Benchmarks on current machine (RTX 5090 Laptop)

Measured with `BODY_INTERM_PRED_LAYERS=0,2`, `--inference_type body`, `--fx 1371`.

| Stage | Time |
|-------|------|
| YOLO detection + FOV + preprocessing | ~54 ms |
| DINOv3 backbone (TRT, batch 3) | ~81 ms (but amortised across 3 crops) |
| Body decoder (compiled, 2 intermediate layers) | ~54 ms |
| PostprocessIK | ~5 ms |
| **Total per frame** | **~68 ms → ~14-15 fps** |

Per-crop backbone time: batch1=30ms, batch2=57ms (1.87× over PyTorch baseline).

---

## Settings that were tested and abandoned

| Setting | Result |
|---------|--------|
| `MHR_USE_CUDA_GRAPH=1` | 4.84 fps — slower. CUDA graph tree conflicts with `torch.compile`. |
| `BODY_INTERM_PRED_LAYERS=0,1,2` | Full accuracy but ~15ms slower per frame. |
| `IMG_SIZE=384` (without TRT backbone) | 6.5 fps — best config without TRT backbone. |
| `IMG_SIZE=384` (with TRT backbone for 512) | Not compatible — TRT engine built for 512 fixed. Would need engine rebuild. |
| `USE_COMPILE=1` backbone without TRT | 4.85 fps at 512px, 6.5fps at 384px. |
| `PARALLEL_DECODERS=1` | 4.84 fps — slower than serial. |
