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
| `markers_<name>.trc` | OpenSim TRC marker file — markers in mm, Y-up (39 body / 79 full mode) |
| `markers_<name>_ik.mot` | OpenSim IK-solved joint angles — 40 DOF, degrees |
| `markers_<name>_model.osim` | Pose2Sim Wholebody body model (scaled, used by IK solver) |
| `markers_<name>_mesh.glb` | Animated full-body mesh GLB + skeleton overlay (skip with `--no_mesh_glb`) |
| `markers_<name>_anatomical.glb` | OpenSim anatomical bones animated by IK motion |
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
| `fast_sam_3d_body` conda env | 3D pose inference + GLB export | See [SETUP.md](SETUP.md) |
| `opensim` conda env | Scale Tool + IK solver | `conda create -n opensim python=3.10 && conda install -n opensim -c opensim-org opensim` |

The pipeline detects missing dependencies and falls back gracefully:
- No `opensim` env → IK and anatomical GLB skipped, TRC and mesh GLB still written

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
| `--inference_type` | `body` | `body` = faster, fewer markers · `full` = body + hands (73 markers, IK-ready) |
| `--person_height` | `1.75` | Known subject height in metres. Scales 3D output to match. |
| `--subject_mass` | `70.0` | Subject mass in kg (used for model scaling only, not kinematics) |
| `--target_fps` | `30` | Process at this FPS by skipping frames (0 = every frame) |
| `--max_frames` | `0` | Stop after this many input frames (0 = full video) |
| `--no_mesh_glb` | off | Skip full body mesh + anatomical GLB export |
| `--fy` | same as `--fx` | Focal length y if different from fx |
| `--cx`, `--cy` | frame centre | Principal point in pixels. Defaults to width/2, height/2. |
| `--local_checkpoint` | `./checkpoints/sam-3d-body-dinov3` | Path to SAM-3D-Body checkpoint directory |

### Multi-person flags

| Flag | Default | Description |
|------|---------|-------------|
| `--multi_person` | off | Enable multi-person tracking and per-person export |
| `--tracker` | `botsort` | Multi-object tracker: `botsort` (re-ID after occlusion), `bytetrack` (lighter, no re-ID), `none` (legacy centroid matching) |
| `--person_heights` | — | Comma-separated heights in metres, assigned left-to-right as seen in the video. E.g. `--person_heights 1.69,1.82`. Overrides `--person_height`. |
| `--max_persons` | `6` | Maximum person detections per frame (keeps largest bboxes) |
| `--run_ik_per_person` | off | Run OpenSim Scale + IK for each tracked person (slow) |

Multi-person mode produces additional output files:

| File | Description |
|------|-------------|
| `markers_<name>_person01.trc` | Per-person TRC (one per tracked person, sorted left-to-right) |
| `markers_<name>_person01_mesh.glb` | Per-person mesh GLB with skeleton overlay |
| `markers_<name>_combined.trc` | All persons in one TRC with world-space offsets |
| `markers_<name>_combined_mesh.glb` | All persons in one GLB scene (distinct translucent colors) |

### Lean / floor correction flags

| Flag | Default | Description |
|------|---------|-------------|
| `--floor_moge` | off | Estimate floor plane from MoGe depth on frame 0. Uses camera-pitch angle to correct forward lean from tilted cameras. |
| `--lean_ref_frame` | — | Frame index (0-based) where the person is standing upright. The spine lean measured over a ±5 frame window is used as the correction for all frames. Combines with `--floor_moge`. |
| `--lean_angle` | — | Manual lean correction in degrees. Positive tilts backward (corrects forward lean). Overrides auto-detection. |
| `--no_lean_fix` | off | Skip all automatic forward-lean correction |
| `--lean_cam_pitch_fix` | off | Experimental camera-pitch-based lean correction |

### Detection tuning flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bbox_thr` | `0.5` | Detection confidence threshold |
| `--nms_thr` | `0.3` | NMS threshold |
| `--detect_then_infer` | off | Run detector first, then send detected boxes to pose estimator in chunks |
| `--inference_batch_cap` | `4` | Max person crops per estimator batch (avoids TRT profile limits) |
| `--force_bboxes` | off | Always run per-detection bbox inference |

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

## Multi-person example

```bash
python demo_video_opensim.py \
    --video_path ./videos/two_people_squat.mp4 \
    --inference_type body \
    --fx 1371 \
    --multi_person \
    --person_heights 1.69,1.82 \
    --floor_moge \
    --lean_ref_frame 0
```

This tracks both people with BoT-SORT, exports per-person TRC + mesh GLB files,
a combined TRC and a combined mesh GLB with both people in one scene.
Person order is left-to-right as seen in the video.

---

## Notes on the first run (warm-up)

When `USE_COMPILE=1` or `DECODER_COMPILE=1` is set, `torch.compile` triggers a JIT compilation on the first few frames. This takes 30–60 seconds and shows slower per-frame times. Subsequent frames run at full speed. The compiled kernels are cached by PyTorch and reused across runs as long as the environment has not changed.

When `USE_TRT_BACKBONE=1` is set, the TRT engine is loaded from disk (~1.6 GB). This takes ~5 seconds at startup but inference is faster.
