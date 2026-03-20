#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fast SAM 3D Body – OpenSim Video Export
========================================
Matches SAM3D-OpenSim output format exactly.  Writes to <output_dir>/:

  markers_<name>_skeleton.mp4   — annotated video with 2D skeleton overlay
  markers_<name>.trc            — 73 markers in mm (Y-up, OpenSim coords)
  markers_<name>_ik.mot         — joint angles from OpenSim IK solver (40 DOF)
  markers_<name>_model.osim     — Pose2Sim_Simple body model for IK
  markers_<name>.glb            — animated skeleton GLB (Blender/rigify)
  markers_<name>_mesh.glb       — full body mesh GLB (--no_mesh_glb to skip)
  inference_meta.json           — video metadata
  video_outputs.json            — per-frame raw 3D keypoints
  processing_report.json        — pipeline summary and timings

Usage:
    conda activate fast_sam_3d_body

    SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 \\
    USE_TRT_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 COMPILE_MODE=reduce-overhead \\
    MHR_NO_CORRECTIVES=1 GPU_HAND_PREP=1 BODY_INTERM_PRED_LAYERS=0,2 \\
    DEBUG_NAN=0 PARALLEL_DECODERS=0 COMPILE_WARMUP_BATCH_SIZES=1 \\
    python demo_video_opensim.py \\
        --video_path ./videos/aitor_garden_walk.mp4 \\
        --detector yolo_pose \\
        --detector_model checkpoints/yolo/yolo11m-pose.engine \\
        --fx 1371

Coordinate system (TRC)
------------------------
  OpenSim Y-up:  X = forward (anterior), Y = up, Z = right (lateral)
  Units: mm (millimetres) — matches SAM3D-OpenSim convention.

Post-processing pipeline
-------------------------
  1. PostProcessor        — missing-frame interpolation, bone normalisation,
                             Butterworth 6 Hz low-pass filter
  2. CoordinateTransformer — camera → OpenSim Y-up axes, height scaling,
                              per-frame ground alignment
  3. KeypointConverter    — MHR70 → 73 OpenSim marker names
                             (body + hands + derived PelvisCenter/Thorax/SpineMid)
  4. TRCExporter          — writes .trc in mm
  5. OpenSim IK           — runs InverseKinematicsTool via opensim conda env
                             → produces _ik.mot (40 DOF) and _ik_marker_errors.sto
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.export.opensim_exporter import (
    write_skeleton_glb,
    write_mesh_glb,
)
from sam_3d_body.export.post_processing import PostProcessor
from sam_3d_body.export.coordinate_transform import CoordinateTransformer
from sam_3d_body.export.keypoint_converter import KeypointConverter
from sam_3d_body.export.trc_exporter import TRCExporter
from sam_3d_body.export.opensim_ik_runner import run_ik

# Pose2Sim model template bundled with the repo
_MODEL_TEMPLATE = os.path.join(parent_dir, "assets", "pose2sim_simple_model.osim")

# Blender rig and export script for GLB generation
_BLEND_RIG    = os.path.join(parent_dir, "assets", "Import_OS4_Patreon_Aitor_Skely.blend")
_GLB_SCRIPT   = os.path.join(parent_dir, "export_glb_skely.py")


def _run_blender_glb(mot_path: str, output_path: str, trc_path: str | None, fps: float) -> bool:
    """Run Blender headless to export a GLB from an IK MOT file.

    Returns True on success, False if Blender or the rig file is unavailable.
    """
    blender = shutil.which("blender")
    if blender is None:
        print("  [GLB] blender not found in PATH — skipping Blender GLB export")
        return False
    if not os.path.isfile(_BLEND_RIG):
        print(f"  [GLB] rig file not found: {_BLEND_RIG}")
        return False
    if not os.path.isfile(_GLB_SCRIPT):
        print(f"  [GLB] export script not found: {_GLB_SCRIPT}")
        return False

    cmd = [
        blender, "--background", _BLEND_RIG,
        "--python", _GLB_SCRIPT,
        "--",
        "--mot", mot_path,
        "--output", output_path,
        "--fps", str(int(round(fps))),
    ]
    if trc_path and os.path.isfile(trc_path):
        cmd += ["--trc", trc_path]

    print(f"  [GLB] Running Blender GLB export...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [GLB] Blender exited with code {result.returncode}")
        print(result.stderr[-2000:] if result.stderr else "")
        return False

    if not os.path.isfile(output_path):
        print(f"  [GLB] Blender ran but output file not created: {output_path}")
        return False

    print(f"  [GLB] Blender GLB written: {output_path}")
    return True


def draw_results_on_frame(img_bgr, outputs, visualizer):
    out = img_bgr.copy()
    for person in outputs:
        if "bbox" in person:
            b = person["bbox"]
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        if "pred_keypoints_2d" in person and visualizer is not None:
            kpts = person["pred_keypoints_2d"]
            kpts_with_score = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=-1)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out_rgb = visualizer.draw_skeleton(out_rgb, kpts_with_score)
            out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out


def main(args):
    # Auto-generate timestamped output directory (matches SAM3D-OpenSim convention)
    if args.output_dir is None:
        video_name_raw = os.path.splitext(os.path.basename(args.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output_{timestamp}_{video_name_raw}"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    cam_int = None
    if args.fx is not None:
        fx = args.fx
        fy = args.fy if args.fy is not None else fx
        cx = args.cx if args.cx is not None else 0.0
        cy = args.cy if args.cy is not None else 0.0
        _K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                           dtype=torch.float32).unsqueeze(0).cuda()
        cam_int = _K
        print(f"Using fixed intrinsics: fx={fx:.1f} fy={fy:.1f}")

    # ── Model loading ─────────────────────────────────────────────────────────
    print("Loading SAM 3D Body model...")
    t_load = time.time()
    estimator = setup_sam_3d_body(
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
    )
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    print(f"Model loaded in {time.time() - t_load:.1f}s")

    # ── Video I/O ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames > 0:
        total = min(total, args.max_frames)

    frame_step = max(1, round(fps / args.target_fps)) if args.target_fps > 0 else 1
    out_fps    = fps / frame_step

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    prefix = f"markers_{video_name}"

    # Output paths — mirror SAM3D-OpenSim naming convention
    vid_path       = os.path.join(args.output_dir, f"{prefix}_skeleton.mp4")
    trc_path       = os.path.join(args.output_dir, f"{prefix}.trc")
    ik_mot_path    = os.path.join(args.output_dir, f"{prefix}_ik.mot")
    errors_path    = os.path.join(args.output_dir, "_ik_marker_errors.sto")
    osim_path      = os.path.join(args.output_dir, f"{prefix}_model.osim")
    skel_glb       = os.path.join(args.output_dir, f"{prefix}.glb")
    mesh_glb       = os.path.join(args.output_dir, f"{prefix}_mesh.glb")
    meta_path      = os.path.join(args.output_dir, "inference_meta.json")
    outputs_path   = os.path.join(args.output_dir, "video_outputs.json")

    writer = cv2.VideoWriter(
        vid_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )

    print(f"\nVideo: {width}x{height} @ {fps:.1f}fps | {total} frames")
    print(f"Frame step: {frame_step} | Output: {args.output_dir}\n")

    t_start = time.time()

    # ── Per-frame processing ──────────────────────────────────────────────────
    frame_idx       = 0
    processed       = 0
    timestamps      = []
    all_kpts_raw    = []   # [N_frames] of [70, 3] camera-space kpts, or None
    all_cam_t       = []   # [N_frames] of [3], or None
    all_verts       = []   # [N_frames] of [18439, 3] or None  (for mesh GLB)
    all_raw_outputs = []   # for video_outputs.json
    inference_times = []

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_cam_int = None
        if cam_int is not None:
            frame_cam_int = cam_int.clone()
            if args.cx is None:
                frame_cam_int[0, 0, 2] = width / 2.0
            if args.cy is None:
                frame_cam_int[0, 1, 2] = height / 2.0

        t0 = time.time()
        try:
            outputs = estimator.process_one_image(
                frame_rgb,
                hand_box_source=args.hand_box_source,
                inference_type=args.inference_type,
                cam_int=frame_cam_int,
            )
        except Exception as e:
            print(f"  Frame {frame_idx}: inference error — {e}")
            writer.write(frame_bgr)
            timestamps.append(frame_idx / fps)
            all_kpts_raw.append(None)
            all_cam_t.append(None)
            all_verts.append(None)
            all_raw_outputs.append({"frame": f"frame_{frame_idx:06d}.jpg", "outputs": []})
            frame_idx += 1
            processed += 1
            continue

        inf_t = time.time() - t0
        inference_times.append(inf_t)

        # Pick the largest person (most confident detection)
        person = outputs[0] if outputs else None

        # Collect raw keypoints and camera translation for this frame
        if person is not None:
            kpts  = person.get("pred_keypoints_3d")   # [70, 3] camera space
            cam_t = person.get("pred_cam_t")           # [3]
            if kpts is not None and cam_t is not None and not np.any(np.isnan(kpts)):
                all_kpts_raw.append(kpts.copy())
                all_cam_t.append(cam_t.copy())
            else:
                all_kpts_raw.append(None)
                all_cam_t.append(None)
        else:
            all_kpts_raw.append(None)
            all_cam_t.append(None)

        timestamps.append(frame_idx / fps)

        # Collect raw outputs for video_outputs.json
        frame_out = {"frame": f"frame_{frame_idx:06d}.jpg", "outputs": []}
        for p in outputs:
            entry: dict = {}
            if "bbox" in p:
                entry["bbox"] = [float(x) for x in p["bbox"]]
            if "pred_cam_t" in p and p["pred_cam_t"] is not None:
                entry["focal_length"] = float(p.get("focal_length", 0.0))
            if "pred_keypoints_3d" in p and p["pred_keypoints_3d"] is not None:
                entry["pred_keypoints_3d"] = p["pred_keypoints_3d"].tolist()
            frame_out["outputs"].append(entry)
        all_raw_outputs.append(frame_out)

        # Collect mesh vertices for mesh GLB
        if not args.no_mesh_glb and person is not None:
            verts = person.get("pred_vertices")
            cam_t = person.get("pred_cam_t")
            if verts is not None and cam_t is not None and not np.any(np.isnan(verts)):
                v_world = verts + cam_t[None, :]
                all_verts.append(v_world.astype(np.float32))
            else:
                all_verts.append(None)
        else:
            all_verts.append(None)

        vis_frame = draw_results_on_frame(frame_bgr, outputs, visualizer)
        writer.write(vis_frame)

        processed += 1
        avg_fps = 1.0 / (sum(inference_times) / len(inference_times))
        detected = len(outputs)
        print(
            f"  [{processed}/{total // frame_step}] frame {frame_idx:5d} | "
            f"{inf_t:.2f}s | {detected} person(s) | avg {avg_fps:.2f} fps",
            flush=True,
        )

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nDone! {processed} frames processed.")
    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print(f"Avg inference: {avg:.2f}s/frame ({1/avg:.2f} fps)")

    # ── Build raw keypoint array (NaN for missing frames) ─────────────────────
    N = len(timestamps)
    kpts_stack  = np.full((N, 70, 3), np.nan, dtype=np.float64)
    cam_t_stack = np.full((N, 3),     np.nan, dtype=np.float64)
    for i, (k, t_) in enumerate(zip(all_kpts_raw, all_cam_t)):
        if k is not None and t_ is not None:
            kpts_stack[i]  = k
            cam_t_stack[i] = t_

    good = int(np.sum(~np.any(np.isnan(kpts_stack), axis=(1, 2))))
    print(f"\n  Frames with detected person: {good}/{processed}")

    if good == 0:
        print("  No valid frames — skipping OpenSim export.")
        return

    # ── Post-processing pipeline ───────────────────────────────────────────────
    subject_height = args.person_height if args.person_height is not None else 1.75

    print("\nPost-processing keypoints...")

    # 1. Interpolate missing frames, normalise bone lengths, Butterworth filter
    post_proc = PostProcessor()
    kpts_processed = post_proc.process(kpts_stack, fps=out_fps, subject_height=subject_height)

    # 2. Rotate axes (camera → OpenSim Y-up), scale to subject height,
    #    center pelvis horizontally, align feet to Y=0 each frame
    transformer = CoordinateTransformer(subject_height=subject_height)
    kpts_opensim = transformer.transform(
        kpts_processed,
        center_pelvis=True,
        align_to_ground=True,
        apply_global_translation=False,
    )

    # 2b. Correct systematic forward lean (matches SAM3D-OpenSim default)
    kpts_opensim = transformer.correct_forward_lean(kpts_opensim)

    # 3. Map MHR70 → 73 OpenSim marker names (body + hands + derived)
    body_only = (args.inference_type == "body")
    converter = KeypointConverter()
    markers_array, marker_names = converter.convert(
        kpts_opensim, include_derived=True, body_only=body_only
    )

    # Body-only markers for GLB skeleton visualisation (27 markers, fixed links)
    markers_body, _ = converter.convert(
        kpts_opensim, include_derived=True, body_only=True
    )

    # ── Export OpenSim files ──────────────────────────────────────────────────
    print("\nExporting OpenSim files...")

    # Save metadata JSONs
    inference_time = time.time() - t_start
    meta = {
        "input_video": os.path.abspath(args.video_path),
        "fps": fps,
        "num_frames": total,
        "video_info": {
            "fps": fps,
            "frame_count": total,
            "width": width,
            "height": height,
            "duration": total / fps,
        },
        "inference_time": inference_time,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(outputs_path, "w") as f:
        json.dump(all_raw_outputs, f)

    print(f"  Writing TRC (mm)  → {trc_path}")
    exporter = TRCExporter(fps=out_fps, units="mm")
    exporter.export(markers_array, marker_names, trc_path)

    # Copy the Pose2Sim body model (needed by OpenSim IK)
    if os.path.isfile(_MODEL_TEMPLATE):
        shutil.copy(_MODEL_TEMPLATE, osim_path)
        print(f"  Writing model     → {osim_path}")
    else:
        print(f"  WARNING: model template not found at {_MODEL_TEMPLATE}")

    frames_markers      = [markers_array[i] for i in range(N)]
    frames_markers_body = [markers_body[i]  for i in range(N)]

    print(f"  Running OpenSim IK → {ik_mot_path}")
    ik_ok = run_ik(
        model_path=osim_path,
        trc_path=trc_path,
        mot_path=ik_mot_path,
        errors_path=errors_path,
    )
    if not ik_ok:
        print("  WARNING: OpenSim IK failed or opensim env not found.")

    print(f"  Writing skeleton GLB → {skel_glb}")
    blender_glb_ok = False
    if ik_ok:
        blender_glb_ok = _run_blender_glb(
            mot_path=ik_mot_path,
            output_path=skel_glb,
            trc_path=trc_path,
            fps=out_fps,
        )
    if not blender_glb_ok:
        print("  [GLB] Falling back to built-in skeleton GLB writer")
        write_skeleton_glb(skel_glb, timestamps, frames_markers_body)

    if not args.no_mesh_glb:
        print(f"  Writing mesh GLB  → {mesh_glb}")
        write_mesh_glb(mesh_glb, timestamps, all_verts, estimator.faces)

    # Write processing report (matches SAM3D-OpenSim convention)
    report_path = os.path.join(args.output_dir, "processing_report.json")
    total_time = time.time() - t_start
    report = {
        "input": os.path.abspath(args.video_path),
        "output_dir": args.output_dir,
        "subject": {"height": subject_height},
        "video_info": {"fps": fps, "frame_count": total, "width": width, "height": height},
        "processing": {
            "fps": out_fps,
            "num_frames": processed,
            "num_markers": len(marker_names),
            "ik_success": ik_ok,
            "glb_blender": blender_glb_ok,
        },
        "timings": {"total": total_time},
        "outputs": {
            "video": vid_path,
            "trc": trc_path,
            "mot": ik_mot_path if ik_ok else None,
            "model": osim_path,
            "glb": skel_glb,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOutput folder: {args.output_dir}")
    print("\nOutput files:")
    print(f"  Video:               {os.path.basename(vid_path)}")
    print(f"  TRC (73 markers/mm): {os.path.basename(trc_path)}")
    print(f"  IK MOT (40 DOF):     {os.path.basename(ik_mot_path)}" + (" ✓" if ik_ok else " (skipped)"))
    print(f"  Body model:          {os.path.basename(osim_path)}")
    glb_note = " (Blender)" if blender_glb_ok else " (built-in)"
    print(f"  Skeleton GLB:        {os.path.basename(skel_glb)}{glb_note}")
    if not args.no_mesh_glb:
        print(f"  Mesh GLB:            {os.path.basename(mesh_glb)}")
    print(f"  Processing report:   {os.path.basename(report_path)}")

    print("""
─────────────────────────────────────────────────────────────────
OpenSim workflow:
  1. Load markers_output_<name>_model.osim in OpenSim
  2. Scale Tool → use TRC for static pose calibration
  3. IK Tool → load TRC → IK MOT is already written if opensim env found
  4. GLB files can be previewed in Blender (File → Import → glTF 2.0).
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast SAM 3D Body – OpenSim Export")
    parser.add_argument("--video_path", default="./videos/aitor_garden_walk.mp4")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Default: auto-generated as "
                             "output_YYYYMMDD_HHMMSS_<videoname> next to the video.")
    parser.add_argument("--detector", default="yolo_pose")
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--hand_box_source", default="yolo_pose")
    parser.add_argument("--local_checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    parser.add_argument("--hands", action="store_true",
                        help="(legacy) Equivalent to --inference_type full")
    parser.add_argument("--inference_type", default=None, choices=["full", "body"],
                        help="'full' includes hands (default, needed for IK), "
                             "'body' skips hands (faster but IK hand markers missing).")
    parser.add_argument("--target_fps", type=float, default=0,
                        help="Process at this FPS (0=all frames)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many input frames (0=all)")
    parser.add_argument("--no_mesh_glb", action="store_true",
                        help="Skip full body mesh GLB export (saves ~185 MB for long videos)")
    parser.add_argument("--person_height", type=float, default=None,
                        help="Known person height in metres (e.g. 1.69). Scales all 3D output "
                             "so the skeleton height matches this value.")
    parser.add_argument("--floor_level", action="store_true",
                        help="(Legacy flag) Per-frame ground alignment is always applied.")
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length x (pixels). Skips MoGe FOV estimation if set.")
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point x (pixels). Defaults to frame_width/2.")
    parser.add_argument("--cy", type=float, default=None)
    args = parser.parse_args()
    # Resolve inference_type: explicit --inference_type wins;
    # default is 'full' (hands needed for IK hand markers LIndex3/RIndex3/etc.)
    if args.inference_type is None:
        args.inference_type = "full" if args.hands else "full"
    main(args)
