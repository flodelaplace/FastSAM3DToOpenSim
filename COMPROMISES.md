# Compromises Made to Reach ~15 FPS

The original pipeline without any optimisations runs at approximately 5 fps on an RTX 5090 Laptop.
The current configuration achieves ~14-15 fps. This document explains every accuracy and
quality trade-off that was made to get there, in order of descending impact.

---

## 1. No hand or finger tracking (`--inference_type body`)

**Speed gain**: ~9.4 fps (largest single gain, measured: 14.65 fps body-only vs 5.24 fps with hands)

**What was cut**: The hand decoder is skipped entirely. This means:
- No wrist-distal keypoints (knuckles, fingertips)
- No finger joint angles in the MOT file
- No finger mesh deformation in the GLB

**What remains**: All 24 body landmarks (shoulders, elbows, wrists, hips, knees, ankles, toes, neck, nose). Wrist positions are still estimated from the body decoder.

**When it matters**: Only if you need detailed hand motion capture — e.g., for hand therapy, sign language, or finger kinematics. For gait analysis, sports biomechanics, or general body motion, the body decoder is fully sufficient.

**To undo**: Change `--inference_type body` to `--inference_type full`.

---

## 2. Fewer intermediate decoder prediction layers (`BODY_INTERM_PRED_LAYERS=0,2`)

**Speed gain**: ~15 ms/frame (~1-2 fps)

**What was cut**: The body decoder refines its pose estimate iteratively at 3 intermediate layers (0, 1, 2). Skipping layer 1 saves one full forward pass of the prediction head. The final pose is predicted at layer 2 either way, but layer 1's refinement contribution is lost.

**What it affects**: Accuracy of joint localisation in difficult cases — heavy occlusion, unusual poses, fast motion blur. In normal walking or standing, the difference is visually imperceptible. Under stress conditions (one leg occluded by the other, arms overhead), the pose may be slightly less well-aligned.

**Full accuracy command**: change `BODY_INTERM_PRED_LAYERS=0,2` to `BODY_INTERM_PRED_LAYERS=0,1,2`.

---

## 3. No corrective blend shapes (`MHR_NO_CORRECTIVES=1`)

**Speed gain**: ~3-5 ms/frame

**What was cut**: Corrective blend shapes are small per-joint mesh deformations applied on top of the base skinning. They model soft tissue compression and stretching that linear blend skinning cannot represent — for example, the bulge of a bicep flexing or the crease of the elbow when bent.

**What it affects**: The mesh surface quality only. Joint positions and skeleton keypoints are completely unaffected. The mesh looks like standard linear blend skinning — acceptable for OpenSim markers and GLB preview, but slightly cartoon-like at extreme joint angles.

**When it matters**: Only for high-quality mesh rendering or close-up mesh inspection. Irrelevant for TRC/MOT data.

**To undo**: Remove `MHR_NO_CORRECTIVES=1`.

---

## 4. Fast, small depth estimation model (`FOV_MODEL=s FOV_LEVEL=0`)

**Speed gain**: ~5-8 ms/frame for the MoGe component

**What was cut**: MoGe can run in a large accurate mode or a small fast mode. Using `FOV_MODEL=s FOV_LEVEL=0` picks the smallest model at the coarsest resolution.

**What it affects**:
- If `--fx` is provided (as in the run command), MoGe does not estimate the focal length — that value is taken directly from the flag. So `FOV_MODEL=s` only affects the depth conditioning used internally for 3D lifting.
- If `--fx` is NOT provided, the FOV estimation will be less precise, meaning the 3D world scale (metres per pixel) will be slightly off. This makes the absolute depth (Z position) and the height of the person inaccurate, though relative joint angles are unaffected.

**Recommendation**: always provide `--fx` from known camera calibration. When `--fx` is set, `FOV_MODEL=s` has very little impact on output accuracy.

**To undo**: change `FOV_MODEL=s` to `FOV_MODEL=l` and `FOV_LEVEL=0` to `FOV_LEVEL=2`.

---

## 5. Skip keypoint prompt conditioning (`SKIP_KEYPOINT_PROMPT=1`)

**Speed gain**: ~2-3 ms/frame

**What was cut**: The YOLO detector outputs 17 2D body keypoints along with the bounding box. Normally these are fed as a conditioning prompt into the decoder to help anchor joint estimates. Skipping this removes one small prompt-construction step.

**What it affects**: The decoder relies more on its own learned priors rather than explicit 2D keypoint anchors. In normal frontal views with clear body visibility, the difference is negligible. In difficult views (back-facing, heavily side-on, partial crops), the pose may drift slightly.

**To undo**: Remove `SKIP_KEYPOINT_PROMPT=1`.

---

## 6. Fixed 512×512 backbone (TRT engine built for this size)

**Speed gain**: this is not itself a loss — it is a constraint imposed by TRT compilation.

**What was cut**: The TRT backbone engine was built for exactly 512×512 input. You cannot change the input image size at runtime without rebuilding the engine. The standard PyTorch backbone supports any resolution.

**What it affects**:
- The 512×512 size is already the default and optimal size for this model.
- If you wanted to test lower resolution (e.g., 384×384, which gives ~6.5 fps with PyTorch backbone and is more accurate per compute-dollar), you need to rebuild the TRT engine for that size.
- 384px was tested without TRT and gave 6.5 fps. With TRT at 384px (not yet built), the estimate is 7-8 fps.

---

## 7. No real-time streaming to OpenSim (`run_publisher.py` not set up)

**Not a speed compromise — a missing data file constraint.**

The `run_publisher.py` script uses the mhr2smpl pipeline to convert MHR mesh predictions to SMPL body parameters and streams them via ZMQ to a live OpenSim session at 50 Hz with pose interpolation.

This requires:
- `mhr2smpl/data/SMPL_NEUTRAL.pkl` — from https://smpl-x.is.tue.mpg.de/ (free academic registration)
- `mhr2smpl/data/mhr2smpl_mapping.npz` — from the MHR repository at `tools/mhr_smpl_conversion/assets/`

Without these files, the offline export pipeline (`demo_video_opensim.py`) still works fully, but real-time streaming is blocked.

---

## Summary table

| Compromise | Speed gain | Accuracy impact | Reversible? |
|-----------|-----------|----------------|------------|
| Skip hand decoder | ~2-3 fps | No hands/fingers | Yes — `--inference_type full` |
| Fewer intermediate layers | ~1-2 fps | Minor pose accuracy in hard cases | Yes — `BODY_INTERM_PRED_LAYERS=0,1,2` |
| No corrective blend shapes | ~0.5 fps | Mesh shape only, not joints | Yes — remove `MHR_NO_CORRECTIVES=1` |
| Small/fast FOV model | ~0.5 fps | Negligible if `--fx` is set | Yes — `FOV_MODEL=l FOV_LEVEL=2` |
| Skip keypoint prompt | ~0.2 fps | Minor, mainly in hard views | Yes — remove `SKIP_KEYPOINT_PROMPT=1` |
| Fixed 512px TRT engine | Enables TRT (+5 fps) | None (optimal size anyway) | Rebuild engine for other sizes |

---

## What was NOT compromised

These are pure engineering optimisations with no accuracy cost:

- **TRT YOLO engine**: same model weights, just faster execution
- **TRT DINOv3 backbone**: same weights, FP16 (negligible precision loss vs FP32)
- **`torch.compile` on decoders**: same computation, optimised kernel scheduling
- **GPU-side crop preprocessing**: same crops, just faster
- **`DEBUG_NAN=0`**: only affects error detection speed, not computation
- **`COMPILE_WARMUP_BATCH_SIZES=1`**: only affects startup time, not steady-state inference
