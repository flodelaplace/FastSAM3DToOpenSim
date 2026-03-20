# How to Run the Pipeline

## Quick start

```bash
conda activate fast_sam_3d_body
cd FastSAM3DToOpenSim

SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 \
USE_TRT_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 COMPILE_MODE=reduce-overhead \
MHR_NO_CORRECTIVES=1 GPU_HAND_PREP=1 BODY_INTERM_PRED_LAYERS=0,2 \
DEBUG_NAN=0 PARALLEL_DECODERS=0 COMPILE_WARMUP_BATCH_SIZES=1 \
python demo_video_opensim.py \
    --video_path ./videos/your_video.mp4 \
    --detector yolo_pose \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --inference_type body \
    --fx 1371
```

This runs at ~14-15fps on RTX 5090 Laptop and writes four output files to `./output_opensim/`.

---

## Output files

All files land in `--output_dir` (default `./output_opensim/`).

| File | Description |
|------|-------------|
| `<video_name>_skeleton.mp4` | Annotated video with 2D skeleton overlay |
| `<video_name>.trc` | OpenSim Track Row Column marker file — 24 body landmarks in metres, Y-up |
| `<video_name>.mot` | OpenSim motion file — 15 anatomical joint angles in degrees (geometric estimate) |
| `<video_name>_skeleton.glb` | Animated skeleton for Blender / three.js (skeletal skinning, ~340 KB for 1136 frames) |
| `<video_name>_mesh.glb` | Animated full body mesh (optional, only with `--mesh_glb`, large file) |

---

## OpenSim workflow after running

1. **Scale model**: `Tools → Scale → load your .osim model` — use the TRC from a static standing pose
2. **Inverse Kinematics**: `Tools → Inverse Kinematics → load TRC` — produces a properly solved MOT
3. The MOT written by this script uses geometric angle estimation (no SMPL). It is useful for preview and sanity-checking but IK output will be more accurate.
4. GLB files: `File → Import → glTF 2.0` in Blender, or drag into three.js / model-viewer.

---

## CLI flags

### Required for real-world use

| Flag | Description |
|------|-------------|
| `--video_path` | Path to input video |
| `--fx` | Camera focal length in pixels. **Strongly recommended.** Skips MoGe depth estimation and uses a pinhole model directly. For a typical phone or webcam, fx ≈ image_width × 1.0 to 1.5. If omitted, MoGe estimates it (adds ~15ms/frame, less accurate). |

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `./output_opensim` | Where to write all output files |
| `--detector` | `yolo_pose` | Person detector backend |
| `--detector_model` | `./checkpoints/yolo/yolo11m-pose.engine` | Path to YOLO model (`.engine` for TRT, `.pt` for PyTorch) |
| `--hands` | off | Include hand and finger tracking. Measured: **5.2 fps** vs **14.7 fps** body-only — nearly 3× slower. |
| `--inference_type` | `body` | Power-user override: `full` = hands, `body` = no hands. Superseded by `--hands` if both given. |
| `--target_fps` | `0` | Process at this FPS by skipping frames (0 = every frame) |
| `--max_frames` | `0` | Stop after this many input frames (0 = full video) |
| `--mesh_glb` | off | Also write a full body mesh GLB (large: ~185 MB for 1136 frames) |
| `--fy` | same as `--fx` | Focal length y if different from fx |
| `--cx`, `--cy` | frame centre | Principal point in pixels. Defaults to width/2, height/2. |
| `--local_checkpoint` | `./checkpoints/sam-3d-body-dinov3` | Path to SAM-3D-Body checkpoint directory |

### Computing focal length (fx)

If you know the camera's field of view:

```
fx = (image_width / 2) / tan(hfov_radians / 2)
```

If you have EXIF data, focal_length_mm and sensor_width_mm:

```
fx = (focal_length_mm / sensor_width_mm) * image_width_pixels
```

For example, a wide-angle phone video at 2160×3840 typically gives `--fx 1371`.

---

## Skeleton-only demo (no OpenSim files)

```bash
python demo_video.py \
    --video_path ./videos/your_video.mp4 \
    --detector_model checkpoints/yolo/yolo11m-pose.pt \
    --inference_type body
```

This writes only the annotated MP4 to `./output_video/`.

---

## Processing a subset of frames (fast test)

```bash
# Process only the first 60 frames at 5fps
python demo_video_opensim.py \
    --video_path ./videos/your_video.mp4 \
    --inference_type body \
    --fx 1371 \
    --max_frames 60 \
    --target_fps 5
```

---

## Notes on the first run (warm-up)

When `USE_COMPILE=1` or `DECODER_COMPILE=1` is set, `torch.compile` triggers a JIT compilation on the first few frames. This takes 30-60 seconds and shows slower per-frame times. Subsequent frames run at full speed. The compiled kernels are cached by PyTorch and reused across runs as long as the environment has not changed.

When `USE_TRT_BACKBONE=1` is set, the TRT engine is loaded from disk (~1.6 GB). This takes ~5 seconds at startup but inference is faster.
