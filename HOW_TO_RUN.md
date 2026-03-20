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
    --inference_type full \
    --fx 1371
```

`--inference_type full` includes hand tracking (required for full 73-marker IK).
Use `--inference_type body` for faster runs (~14 fps) without hand markers.

Measured on RTX 5090 Laptop, 848×480 video:

| Mode | Inference FPS |
|------|--------------|
| `body` — no hands | **~14 fps** |
| `full` — body + hands | **~5.3 fps** |

---

## Output files

Each run creates a timestamped folder: `output_YYYYMMDD_HHMMSS_<videoname>/`

| File | Description |
|------|-------------|
| `markers_<name>_skeleton.mp4` | Annotated video with 2D skeleton overlay |
| `markers_<name>.trc` | OpenSim TRC marker file — 73 landmarks in mm, Y-up |
| `markers_<name>_ik.mot` | OpenSim IK-solved joint angles — 40 DOF, degrees |
| `markers_<name>_model.osim` | Pose2Sim_Simple body model (used by IK solver) |
| `markers_<name>.glb` | Animated rigged skeleton GLB (Blender / three.js) |
| `markers_<name>_mesh.glb` | Animated full-body mesh GLB (~185 MB, skip with `--no_mesh_glb`) |
| `_ik_marker_errors.sto` | OpenSim IK marker tracking errors per frame |
| `inference_meta.json` | Video metadata (fps, resolution, frame count) |
| `video_outputs.json` | Per-frame raw 3D keypoints |
| `processing_report.json` | Pipeline summary: timings, marker count, IK/GLB status |

---

## OpenSim workflow after running

The IK MOT is written automatically — OpenSim does not need to be opened to run IK.
The files are already in the correct format to load directly in OpenSim 4.5+:

1. **Load model**: `File → Open Model` → select `markers_<name>_model.osim`
2. **Preview motion**: `File → Load Motion` → select `markers_<name>_ik.mot`
3. **Inspect markers**: `File → Open Motion Capture Data` → select `markers_<name>.trc`

For custom scaling or re-running IK with different settings:
- Scale Tool: load `markers_<name>_model.osim`, use a static standing TRC
- IK Tool: load the TRC; the body model already has the correct MarkerSet

---

## Dependencies for the full pipeline

| Component | Used for | Setup |
|-----------|----------|-------|
| `fast_sam_3d_body` conda env | 3D pose inference | See [SETUP.md](SETUP.md) |
| `opensim` conda env | IK solver | `conda create -n opensim && conda install -c opensim-org opensim` |
| Blender | Rigged GLB export | `sudo apt install blender && pip3.12 install numpy --break-system-packages` |

The pipeline detects missing dependencies and falls back gracefully:
- No `opensim` env → IK skipped, TRC still written
- No Blender → falls back to built-in skeleton GLB writer

---

## CLI flags

### Required for real-world use

| Flag | Description |
|------|-------------|
| `--video_path` | Path to input video |
| `--fx` | Camera focal length in pixels. **Strongly recommended.** Skips MoGe depth estimation and uses a pinhole model directly. For a typical phone or webcam, fx ≈ image_width × 1.0 to 1.5. If omitted, MoGe estimates it (adds ~15 ms/frame, less accurate). |

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | auto | Output directory. Default: `output_YYYYMMDD_HHMMSS_<videoname>/` in the working directory. |
| `--detector` | `yolo_pose` | Person detector backend |
| `--detector_model` | `./checkpoints/yolo/yolo11m-pose.engine` | Path to YOLO model (`.engine` for TRT, `.pt` for PyTorch) |
| `--inference_type` | `full` | `full` = body + hands (73 markers, IK-ready) · `body` = faster, fewer markers |
| `--person_height` | `1.75` | Known subject height in metres. Scales 3D output to match. |
| `--target_fps` | `0` | Process at this FPS by skipping frames (0 = every frame) |
| `--max_frames` | `0` | Stop after this many input frames (0 = full video) |
| `--no_mesh_glb` | off | Skip full body mesh GLB export (saves ~185 MB for long videos) |
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
# Process only the first 60 frames at 5 fps
python demo_video_opensim.py \
    --video_path ./videos/your_video.mp4 \
    --inference_type body \
    --fx 1371 \
    --max_frames 60 \
    --target_fps 5
```

---

## Notes on the first run (warm-up)

When `USE_COMPILE=1` or `DECODER_COMPILE=1` is set, `torch.compile` triggers a JIT compilation on the first few frames. This takes 30–60 seconds and shows slower per-frame times. Subsequent frames run at full speed. The compiled kernels are cached by PyTorch and reused across runs as long as the environment has not changed.

When `USE_TRT_BACKBONE=1` is set, the TRT engine is loaded from disk (~1.6 GB). This takes ~5 seconds at startup but inference is faster.
