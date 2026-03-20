# FastSAM3DToOpenSim

> **OpenSim biomechanics extension of [Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body)**
>
> Takes the Fast-SAM-3D-Body inference pipeline and exports every frame of a video directly
> to OpenSim-ready files: TRC marker trajectories, MOT joint angles, and animated GLB files
> for Blender / three.js.

---

## What this adds on top of Fast-SAM-3D-Body

| Feature | Fast-SAM-3D-Body | This repo |
|---------|-----------------|-----------|
| 3D body mesh inference | ✓ | ✓ |
| Annotated video output | ✓ | ✓ |
| OpenSim TRC marker file | — | ✓ |
| OpenSim MOT joint angle file | — | ✓ |
| Animated skeleton GLB (Blender / three.js) | — | ✓ |
| Animated full-body mesh GLB | — | ✓ (opt-in) |
| `--hands` flag to toggle hand tracking | — | ✓ |
| Full setup guides for Linux + Windows | — | ✓ |
| Per-GPU expected FPS table | — | ✓ |
| TRT engine build instructions | partial | ✓ |

---

## Output files

For each input video, the pipeline writes:

```
output_opensim/
  <video_name>_skeleton.mp4    — annotated video with 2D skeleton overlay
  <video_name>.trc             — OpenSim marker file (24 body landmarks, metres, Y-up)
  <video_name>.mot             — OpenSim motion file (15 joint angles, degrees)
  <video_name>_skeleton.glb    — animated skeleton  (~340 KB for 1136 frames)
  <video_name>_mesh.glb        — animated full-body mesh (opt-in, --mesh_glb)
```

### TRC marker set — 24 landmarks

nose · l/r shoulder · l/r elbow · l/r wrist · l/r hip · l/r knee · l/r ankle ·
l/r big toe · l/r small toe · l/r heel · l/r olecranon · l/r acromion · neck

### MOT joint angles — 15 columns

pelvis translation (tx/ty/tz) · l/r hip flexion · l/r hip adduction ·
l/r knee flexion · l/r ankle dorsiflexion · l/r elbow flexion ·
trunk flexion · trunk lateral lean

> The MOT file uses geometric estimation from 3D landmark positions — no SMPL model needed.
> For production biomechanics, run OpenSim Inverse Kinematics on the TRC file instead;
> that output will be more accurate.

---

## Performance (measured, RTX 5090 Laptop, Linux)

| Mode | FPS |
|------|-----|
| Body only — `--inference_type body` (default) | **14.7 fps** |
| Body + hands — `--hands` | **5.2 fps** |

See [COMPROMISES.md](COMPROMISES.md) for a full breakdown of every trade-off made to reach these numbers.

---

## Quick start

### 1. Install

- **Linux**: [SETUP.md](SETUP.md)
- **Windows**: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

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
    --inference_type body \
    --fx 1371
```

Replace `--fx 1371` with your camera focal length in pixels
(see [HOW_TO_RUN.md](HOW_TO_RUN.md) for how to compute it).
If unknown, omit `--fx` and the pipeline will estimate it from the image.

To include hands (5.2 fps instead of 14.7 fps):

```bash
# add --hands to the command above
python demo_video_opensim.py ... --hands
```

### 3. Open in OpenSim

1. **Scale**: `Tools → Scale → load your .osim model` using a static standing TRC
2. **Inverse Kinematics**: `Tools → Inverse Kinematics → load TRC` → outputs a solved MOT

### 4. Open in Blender

`File → Import → glTF 2.0` → select `_skeleton.glb` or `_mesh.glb`

---

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video_path` | — | Input video |
| `--fx` | auto (MoGe) | Camera focal length in pixels |
| `--inference_type` | `body` | `body` = 14.7 fps · `full` = hands, 5.2 fps |
| `--hands` | off | Shorthand for `--inference_type full` |
| `--mesh_glb` | off | Also write animated full-body mesh GLB |
| `--target_fps` | 0 | Downsample input to this FPS (0 = every frame) |
| `--max_frames` | 0 | Stop after N frames (0 = full video) |
| `--output_dir` | `./output_opensim` | Output directory |

---

## Coordinate system

All 3D outputs use **OpenSim Y-up** convention:

```
X = camera X (lateral, rightward)
Y = −camera Y (vertical, upward)
Z = camera Z (depth, forward into scene)
Units: metres
```

---

## Real-time ZMQ streaming

`run_publisher.py` streams pose to OpenSim live via ZMQ at 50 Hz using the mhr2smpl pipeline.
Requires two additional data files not included in this repo:

- `mhr2smpl/data/SMPL_NEUTRAL.pkl` — from https://smpl-x.is.tue.mpg.de/ (free academic registration)
- `mhr2smpl/data/mhr2smpl_mapping.npz` — from the MHR repo at `tools/mhr_smpl_conversion/assets/`

Without these, the offline export pipeline works fully.

---

## Architecture

```
Input video
    │
    ▼
YOLO v11 pose detector
    │  bounding boxes + 2D keypoints
    ▼
MoGe depth / FOV estimator  (TRT, model=s, level=0)
    │  camera intrinsics + depth conditioning
    ▼
DINOv3-ViT/H backbone  (TRT, 512×512, FP16)
    │  image tokens [B, 1280, 32, 32]
    ▼
MHR body decoder  (torch.compile, 2 intermediate layers)
    │  pred_keypoints_3d [70, 3]
    │  pred_cam_t [3]
    │  pred_vertices [18439, 3]
    ▼
OpenSim exporter  (sam_3d_body/export/opensim_exporter.py)
    ├── write_trc()           →  .trc
    ├── write_mot()           →  .mot
    ├── write_skeleton_glb()  →  _skeleton.glb  (skeletal skinning, O(N × 24))
    └── write_mesh_glb()      →  _mesh.glb      (morph targets, opt-in)
```

---

## Documentation index

| File | Contents |
|------|----------|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | Run commands, all flags, focal length calculation, OpenSim workflow |
| [SETUP.md](SETUP.md) | Linux install, all GPU variants (5090/5070Ti/5070/4090/A3000…), TRT build |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | Windows install, cmd/PowerShell commands, Windows-specific issues |
| [SETTINGS.md](SETTINGS.md) | Every environment variable, TRT engine specs, benchmark table |
| [COMPROMISES.md](COMPROMISES.md) | Every accuracy trade-off made to reach 14.7 fps, with measured numbers |

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
