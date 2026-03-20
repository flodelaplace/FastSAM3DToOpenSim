#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fast SAM 3D Body – OpenSim Video Export
========================================
Runs body-mesh inference on a video and writes:
  <output_dir>/<video_name>_skeleton.mp4   — annotated video (same as demo_video.py)
  <output_dir>/<video_name>.trc            — OpenSim marker file (24 body landmarks)
  <output_dir>/<video_name>.mot            — OpenSim motion file (approx joint angles)
  <output_dir>/<video_name>_skeleton.glb   — animated skeleton GLB (Blender/three.js)
  <output_dir>/<video_name>_mesh.glb       — animated full body mesh (optional, --mesh_glb)

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
        --inference_type body \\
        --fx 1371

Coordinate system (TRC / GLB)
------------------------------
  OpenSim Y-up:   X = cam_X (lateral), Y = -cam_Y (vertical up), Z = cam_Z (depth)
  Units: metres.

Missing SMPL data files
------------------------
The full mhr2smpl → SMPL pipeline requires:
  mhr2smpl/data/SMPL_NEUTRAL.pkl       — from https://smpl-x.is.tue.mpg.de/
  mhr2smpl/data/mhr2smpl_mapping.npz   — from MHR repo:
                                          tools/mhr_smpl_conversion/assets/

Once placed in mhr2smpl/data/, run run_publisher.py for real-time ZMQ streaming.
The MOT file written by this script uses geometric joint-angle estimation (no SMPL).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.export.opensim_exporter import (
    extract_body_markers,
    write_trc,
    write_mot,
    write_skeleton_glb,
    write_mesh_glb,
    cam_to_opensim,
    BODY_MARKER_NAMES,
)


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

    # Output paths
    vid_path    = os.path.join(args.output_dir, f"{video_name}_skeleton.mp4")
    trc_path    = os.path.join(args.output_dir, f"{video_name}.trc")
    mot_path    = os.path.join(args.output_dir, f"{video_name}.mot")
    skel_glb    = os.path.join(args.output_dir, f"{video_name}_skeleton.glb")
    mesh_glb    = os.path.join(args.output_dir, f"{video_name}_mesh.glb")

    writer = cv2.VideoWriter(
        vid_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )

    print(f"\nVideo: {width}x{height} @ {fps:.1f}fps | {total} frames")
    print(f"Frame step: {frame_step} | Output: {args.output_dir}\n")

    # ── Per-frame processing ──────────────────────────────────────────────────
    frame_idx   = 0
    processed   = 0
    timestamps  = []
    all_markers = []   # [N_frames] of [N_markers, 3] or None
    all_verts   = []   # [N_frames] of [18439, 3] or None  (for mesh GLB)
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
            all_markers.append(None)
            all_verts.append(None)
            frame_idx += 1
            processed += 1
            continue

        inf_t = time.time() - t0
        inference_times.append(inf_t)

        # Pick the largest person (most confident detection)
        person = outputs[0] if outputs else None

        # Collect markers for this frame
        markers = extract_body_markers(person) if person else None
        all_markers.append(markers)
        timestamps.append(frame_idx / fps)

        # Collect mesh vertices for optional mesh GLB
        if args.mesh_glb and person is not None:
            verts = person.get("pred_vertices")
            cam_t = person.get("pred_cam_t")
            if verts is not None and cam_t is not None and not np.any(np.isnan(verts)):
                v_world = verts + cam_t[None, :]  # camera space
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

    # ── Export OpenSim files ──────────────────────────────────────────────────
    print("\nExporting OpenSim files...")

    good = sum(1 for m in all_markers if m is not None)
    print(f"  Frames with detected person: {good}/{processed}")

    print(f"  Writing TRC  → {trc_path}")
    write_trc(trc_path, timestamps, all_markers)

    print(f"  Writing MOT  → {mot_path}")
    write_mot(mot_path, timestamps, all_markers)

    print(f"  Writing skeleton GLB → {skel_glb}")
    write_skeleton_glb(skel_glb, timestamps, all_markers)

    if args.mesh_glb:
        print(f"  Writing mesh GLB → {mesh_glb}")
        write_mesh_glb(mesh_glb, timestamps, all_verts, estimator.faces)

    print("\nOutput files:")
    print(f"  Video:         {vid_path}")
    print(f"  TRC (markers): {trc_path}")
    print(f"  MOT (angles):  {mot_path}")
    print(f"  Skeleton GLB:  {skel_glb}")
    if args.mesh_glb:
        print(f"  Mesh GLB:      {mesh_glb}")

    print("""
─────────────────────────────────────────────────────────────────
OpenSim workflow:
  1. Scale: Tools → Scale → load your .osim model, use TRC for static pose
  2. IK:    Tools → Inverse Kinematics → load TRC → outputs a proper MOT
  3. The MOT written here uses geometric estimation (useful for preview).
  4. GLB files can be previewed in Blender (File → Import → glTF 2.0).
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast SAM 3D Body – OpenSim Export")
    parser.add_argument("--video_path", default="./videos/aitor_garden_walk.mp4")
    parser.add_argument("--output_dir", default="./output_opensim")
    parser.add_argument("--detector", default="yolo_pose")
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--hand_box_source", default="yolo_pose")
    parser.add_argument("--local_checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    parser.add_argument("--hands", action="store_true",
                        help="Include hand tracking (slower, ~9-11 fps vs ~14-15 fps without)")
    parser.add_argument("--inference_type", default=None, choices=["full", "body"],
                        help="Override: 'full' includes hands, 'body' skips them. "
                             "Supersedes --hands if both are given.")
    parser.add_argument("--target_fps", type=float, default=0,
                        help="Process at this FPS (0=all frames)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many input frames (0=all)")
    parser.add_argument("--mesh_glb", action="store_true",
                        help="Also export animated full body mesh GLB (large file)")
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length x (pixels). Skips MoGe FOV estimation if set.")
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point x (pixels). Defaults to frame_width/2.")
    parser.add_argument("--cy", type=float, default=None)
    args = parser.parse_args()
    # Resolve inference_type: explicit --inference_type wins; otherwise use --hands
    if args.inference_type is None:
        args.inference_type = "full" if args.hands else "body"
    main(args)
