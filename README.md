# FastSAM3DToOpenSim

> **OpenSim biomechanics extension of [Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body)**
>
> Takes the Fast-SAM-3D-Body inference pipeline and exports every frame of a video directly
> to OpenSim-ready files: TRC marker trajectories, IK-solved MOT joint angles, body model,
> and a rigged animated GLB skeleton for Blender / three.js.
> Matches the output format of [SAM3D-OpenSim](https://github.com/AitorIriondo/SAM3D-OpenSim).

---

## What this adds on top of Fast-SAM-3D-Body

| Feature | Fast-SAM-3D-Body | This repo |
|---------|-----------------|-----------|
| 3D body mesh inference | ✓ | ✓ |
| Annotated video output | ✓ | ✓ |
| OpenSim TRC marker file (73 markers, mm) | — | ✓ |
| OpenSim IK-solved MOT (40 DOF, via OpenSim 4.5) | — | ✓ |
| Pose2Sim_Simple body model | — | ✓ |
| Rigged animated skeleton GLB (Blender / three.js) | — | ✓ |
| Animated full-body mesh GLB | — | ✓ (opt-in) |
| Timestamped output folders | — | ✓ |
| Full setup guides for Linux + Windows | — | ✓ |
| TRT engine build instructions | partial | ✓ |

---

## Output files

Each run creates a timestamped folder: `output_YYYYMMDD_HHMMSS_<videoname>/`

```
output_20260320_173750_myvideo/
  markers_<name>_skeleton.mp4    — annotated video with 2D skeleton overlay
  markers_<name>.trc             — 73 OpenSim markers in mm, Y-up
  markers_<name>_ik.mot          — IK-solved joint angles, 40 DOF, degrees
  markers_<name>_model.osim      — Pose2Sim_Simple body model
  markers_<name>.glb             — rigged animated skeleton (~1.3 MB for 584 frames)
  markers_<name>_mesh.glb        — animated full-body mesh (~126 MB, skip with --no_mesh_glb)
  _ik_marker_errors.sto          — IK marker tracking residuals per frame
  inference_meta.json            — video metadata
  video_outputs.json             — per-frame raw 3D keypoints
  processing_report.json         — pipeline summary: timings, IK/GLB status
```

### TRC marker set — 73 landmarks

**Body (30):** Nose · LEye · REye · LEar · REar · LShoulder · RShoulder · LElbow · RElbow ·
LHip · RHip · LKnee · RKnee · LAnkle · RAnkle · LBigToe · LSmallToe · LHeel ·
RBigToe · RSmallToe · RHeel · RWrist · LWrist · LOlecranon · ROlecranon ·
LCubitalFossa · RCubitalFossa · LAcromion · RAcromion · Neck

**Derived (3):** PelvisCenter · Thorax · SpineMid

**Hands (40):** full finger tracking (20 per hand) — only present with `--inference_type full`

### MOT joint angles — 40 DOF

OpenSim IK-solved via `InverseKinematicsTool` using the Pose2Sim_Simple model.
Columns: pelvis tx/ty/tz/tilt/list/rotation · l/r hip flexion/adduction/rotation ·
l/r knee angle · l/r ankle angle · lumbar extension/bending/rotation ·
arm flex/add/rot · elbow flex · pro/sup · wrist flex/dev (both sides).

---

## Performance (measured, RTX 5090 Laptop, Linux, 848×480)

| Mode | Inference FPS | Total time (19.5 s video) |
|------|:---:|:---:|
| `body` — no hands | **~14 fps** | ~50 s |
| `full` — body + hands (IK-ready) | **~5.3 fps** | ~115 s |

Total time includes inference, post-processing, OpenSim IK, and Blender GLB export.

See [COMPROMISES.md](COMPROMISES.md) for a breakdown of every trade-off.

---

## Quick start

### 1. Install

- **Linux**: [SETUP.md](SETUP.md)
- **Windows**: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

Additional dependencies for the full pipeline:

```bash
# OpenSim 4.5 (IK solver)
conda create -n opensim python=3.10
conda install -n opensim -c opensim-org opensim

# Blender + numpy (rigged GLB export)
sudo apt install blender
pip3.12 install numpy --break-system-packages
```

### 2. Run

```bash
conda activate fast_sam_3d_body

SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 \
USE_TRT_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 COMPILE_MODE=reduce-overhead \
MHR_NO_CORRECTIVES=1 GPU_HAND_PREP=1 BODY_INTERM_PRED_LAYERS=0,2 \
DEBUG_NAN=0 PARALLEL_DECODERS=0 COMPILE_WARMUP_BATCH_SIZES=1 \
python demo_video_opensim.py \
    --video_path ./videos/my_video.mp4 \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --inference_type full \
    --fx 1371
```

Replace `--fx 1371` with your camera focal length in pixels
(see [HOW_TO_RUN.md](HOW_TO_RUN.md) for how to compute it).
If unknown, omit `--fx` and the pipeline will estimate it from the image.

### 3. Open in OpenSim

The IK MOT is written automatically. Load directly without re-running IK:

1. `File → Open Model` → select `markers_<name>_model.osim`
2. `File → Load Motion` → select `markers_<name>_ik.mot`
3. `File → Open Motion Capture Data` → select `markers_<name>.trc` to inspect markers

### 4. Open in Blender

`File → Import → glTF 2.0` → select `markers_<name>.glb` (rigged skeleton) or `markers_<name>_mesh.glb`

---

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video_path` | — | Input video |
| `--fx` | auto (MoGe) | Camera focal length in pixels |
| `--inference_type` | `full` | `full` = body + hands (73 markers, IK-ready) · `body` = faster, fewer markers |
| `--person_height` | `1.75` | Known subject height in metres — scales 3D output |
| `--no_mesh_glb` | off | Skip full-body mesh GLB export (saves ~125 MB) |
| `--target_fps` | 0 | Downsample input to this FPS (0 = every frame) |
| `--max_frames` | 0 | Stop after N frames (0 = full video) |
| `--output_dir` | auto | Output directory (default: `output_YYYYMMDD_HHMMSS_<name>/`) |

---

## Coordinate system (TRC)

All 3D outputs use **OpenSim Y-up** convention:

```
X = forward (anterior)
Y = up (superior)
Z = right (lateral)
Units: millimetres (mm)
```

---

## Post-processing pipeline

```
Camera-space keypoints  (N, 70, 3)
    │
    ▼  PostProcessor
    │    ├─ interpolate missing frames
    │    ├─ normalise bone lengths (anthropometric proportions)
    │    └─ Butterworth low-pass filter  (6 Hz, order 4)
    ▼  CoordinateTransformer
    │    ├─ rotate camera → OpenSim Y-up
    │    ├─ scale to subject height
    │    ├─ centre pelvis at origin (XZ)
    │    ├─ align feet to ground (Y=0) per frame
    │    └─ correct forward lean (auto-detected)
    ▼  KeypointConverter   (MHR70 → 73 OpenSim markers)
    ▼  TRCExporter         → markers_<name>.trc  (mm)
    ▼  OpenSim IK          → markers_<name>_ik.mot  (subprocess → opensim env)
    ▼  Blender GLB         → markers_<name>.glb  (subprocess → blender + rigify rig)
    ▼  write_mesh_glb()    → markers_<name>_mesh.glb
```

---

## Real-time ZMQ streaming

`run_publisher.py` streams pose to OpenSim live via ZMQ at 50 Hz using the mhr2smpl pipeline.
Requires two additional data files not included in this repo:

- `mhr2smpl/data/SMPL_NEUTRAL.pkl` — from https://smpl-x.is.tue.mpg.de/ (free academic registration)
- `mhr2smpl/data/mhr2smpl_mapping.npz` — from the MHR repo at `tools/mhr_smpl_conversion/assets/`

Without these, the offline export pipeline works fully.

---

## Documentation index

| File | Contents |
|------|----------|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | Run commands, all flags, focal length calculation, OpenSim workflow, dependency setup |
| [SETUP.md](SETUP.md) | Linux install, all GPU variants (5090/5070Ti/5070/4090/A3000…), TRT build |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | Windows install, cmd/PowerShell commands, Windows-specific issues |
| [SETTINGS.md](SETTINGS.md) | Every environment variable, TRT engine specs, benchmark table |
| [COMPROMISES.md](COMPROMISES.md) | Every accuracy trade-off made to reach these speeds, with measured numbers |

---

## Citation

If you use this OpenSim extension, please also cite the upstream Fast-SAM-3D-Body paper:

```bibtex
@article{yang2026fastsam3dbody,
  title={Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery},
  author={Yang, Timing and He, Sicheng and Jing, Hongyi and Yang, Jiawei and Liu, Zhijian
          and Zou, Chuhang and Wang, Yue},
  journal={arXiv preprint arXiv:2603.15603},
  year={2026}
}
```

---

---

# Fast SAM 3D Body (upstream)

### Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery

[Timing Yang](http://yangtiming.github.io)<sup>1</sup>, [Sicheng He](https://hesicheng.net)<sup>1</sup>, [Hongyi Jing](https://hongyijing.me)<sup>1</sup>, [Jiawei Yang](https://jiawei-yang.github.io)<sup>1</sup>, [Zhijian Liu](https://zhijianliu.com)<sup>2,3</sup>, [Chuhang Zou](https://zouchuhang.github.io)<sup>4</sup><sup>†</sup>, [Yue Wang](https://yuewang.xyz)<sup>1,3</sup><sup>†</sup>

<sup>1</sup>USC Physical Superintelligence (PSI) Lab &nbsp; <sup>2</sup>University of California, San Diego &nbsp; <sup>3</sup>NVIDIA &nbsp; <sup>4</sup>Meta Reality Labs

<sup>†</sup> Joint corresponding authors

<p align="center">
  <a href="https://arxiv.org/abs/2603.15603">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  &nbsp;
  <a href="https://yangtiming.github.io/Fast-SAM-3D-Body-Page/">
    <img src="https://img.shields.io/badge/🌐_Project-Page-4285F4?style=for-the-badge" alt="Project Page">
  </a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="900">
</p>

> **Speed-accuracy overview of Fast SAM 3D Body.** Top left: Qualitative results on in-the-wild images show our framework preserves high-fidelity reconstruction. Top right: Our method achieves up to a **10.25x** end-to-end speedup over SAM 3D Body and replaces the iterative MHR-to-SMPL bottleneck with a **10,000x** faster neural mapping. Bottom: Our system enables real-time humanoid robot control from a single RGB stream at **~65 ms** per frame on an NVIDIA RTX 5090.

## Abstract

SAM 3D Body (3DB) achieves state-of-the-art accuracy in monocular 3D human mesh recovery, yet its inference latency of several seconds per image precludes real-time application. We present **Fast SAM 3D Body**, a training-free acceleration framework that reformulates the 3DB inference pathway to achieve interactive rates. By decoupling serial spatial dependencies and applying architecture-aware pruning, we enable parallelized multi-crop feature extraction and streamlined transformer decoding. Moreover, to extract the joint-level kinematics (SMPL) compatible with existing humanoid control and policy learning frameworks, we replace the iterative mesh fitting with a direct feedforward mapping, accelerating this specific conversion by over 10,000x. Overall, our framework delivers up to a **10.9x** end-to-end speedup while maintaining on-par reconstruction fidelity, even surpassing 3DB on benchmarks such as LSPET. We demonstrate its utility by deploying Fast SAM 3D Body in a vision-only teleoperation system that enables **real-time humanoid control** and the direct collection of manipulation policies from a single RGB stream.

<p align="center">
  <img src="assets/comparison.png" width="900">
</p>

> **Qualitative comparison.** The original SAM 3D Body (left) and our Fast variant (right) yield visually comparable mesh reconstructions across diverse poses and multi-person scenes on 3DPW and EMDB.

## Getting Started

### Environment

Please refer to [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) for environment setup, or use our setup script:

```bash
bash setup_env.sh
conda activate fast_sam_3d_body
```

### Checkpoints

```
checkpoints/
├── sam-3d-body-dinov3/       # Auto-downloaded from HuggingFace on first run
│   ├── model.ckpt
│   └── assets/
│       └── mhr_model.pt
├── yolo/                     # Place YOLO-Pose weights here
│   ├── yolo11m-pose.pt
│   └── yolo11m-pose.engine   # Generated by convert_yolo_pose_trt.py (optional)
└── moge_trt/                 # Generated by build_tensorrt.sh (optional)
    └── moge_dinov2_encoder_fp16.engine
```

### Run

```bash
# Optimized (torch.compile + TensorRT)
bash run_demo.sh
```

### TensorRT Acceleration (Optional)

```bash
# Convert all models (YOLO-Pose + MoGe encoder + DINOv3 backbone)
bash build_tensorrt.sh

# Or convert individually
python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640 --half
python convert_moge_encoder_trt.py --all
python convert_backbone_tensorrt.py --all
```

All generated engines are stored under `./checkpoints/`.

## Real-World Deployment

For instructions on running the publisher, see [docs/realworld_deployment.md](docs/realworld_deployment.md).

We demonstrate a real-time, vision-only teleoperation system for the Unitree G1 humanoid robot using a single RGB camera, operating at ~65 ms end-to-end latency on an NVIDIA RTX 5090.

<p align="center">
  <img src="assets/teleop_qual.png" width="900">
</p>

> **Humanoid teleoperation.** The system tracks diverse whole-body motions including upper-body gestures (a), body rotations (b-e), walking (f), wide stance (g), single-leg standing (h), squatting (i), and kneeling (j).

<p align="center">
  <img src="assets/teleop_policy.png" width="900">
</p>

> **Humanoid policy rollout.** The robot grasps a box on the table with both hands, squats down, and steps to the right. Achieving 80% task success rate with 40 demonstrations collected via our system.

<p align="center">
  <img src="assets/supp_multiview_v1.png" width="900">
</p>

> **Single-View vs Multi-View.** Multi-view fusion resolves depth ambiguities inherent in single-view reconstruction, producing more accurate SMPL body estimates.

## Citation

```bibtex
@article{yang2026fastsam3dbody,
  title={Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery},
  author={Yang, Timing and He, Sicheng and Jing, Hongyi and Yang, Jiawei and Liu, Zhijian and Zou, Chuhang and Wang, Yue},
  journal={arXiv preprint arXiv:2603.15603},
  year={2026}
}
```

## Acknowledgements

This project builds upon [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (3DB) and [Multi-HMR (MHR)](https://github.com/facebookresearch/MHR). We thank the original authors for releasing their models and codebases, which served as the foundation for our acceleration framework.
