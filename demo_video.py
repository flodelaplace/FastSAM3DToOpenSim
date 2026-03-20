# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fast SAM 3D Body - Video Demo
Processes each frame and writes an output video with 2D skeleton overlay.

Usage:
    python demo_video.py --video_path ./videos/aitor_garden_walk.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body, setup_visualizer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = np.array([0.65098039, 0.74117647, 0.85882353])


def save_ply_for_frame(outputs, faces, ply_dir, frame_idx):
    """Save one PLY per detected person for this frame."""
    paths = []
    for pid, person in enumerate(outputs):
        vertices = person.get("pred_vertices")
        cam_t   = person.get("pred_cam_t")
        if vertices is None or cam_t is None:
            continue
        if np.any(np.isnan(vertices)) or np.any(np.isnan(cam_t)):
            continue
        vertex_colors = np.tile((*LIGHT_BLUE, 1.0), (len(vertices), 1))
        mesh = trimesh.Trimesh(
            vertices=vertices + cam_t[None, :],
            faces=faces,
            vertex_colors=vertex_colors,
        )
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        path = os.path.join(ply_dir, f"frame_{frame_idx:06d}_person_{pid:02d}.ply")
        mesh.export(path)
        paths.append(path)
    return paths


def draw_results_on_frame(img_bgr, outputs, visualizer):
    """Draw bounding boxes, hand boxes, and 2D keypoints on frame."""
    out = img_bgr.copy()

    for i, person in enumerate(outputs):
        # Body bounding box (green)
        if "bbox" in person:
            bbox = person["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Left hand box (blue)
        if "lhand_bbox" in person:
            bbox = person["lhand_bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 0), 2)

        # Right hand box (red)
        if "rhand_bbox" in person:
            bbox = person["rhand_bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 255), 2)

        # 2D keypoints via SkeletonVisualizer (expects RGB, returns RGB)
        if "pred_keypoints_2d" in person and visualizer is not None:
            kpts = person["pred_keypoints_2d"]  # [J, 2] (x, y)
            # append score=1 column → [J, 3]
            kpts_with_score = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=-1)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out_rgb = visualizer.draw_skeleton(out_rgb, kpts_with_score)
            out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    return out


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_ply:
        ply_dir = os.path.join(args.output_dir, "ply")
        os.makedirs(ply_dir, exist_ok=True)

    # ── Camera intrinsics (optional, skips FOV estimation if provided) ──
    cam_int = None
    if args.fx is not None:
        # Build 3×3 K matrix as a [1,3,3] float32 cuda tensor
        fx, fy = args.fx, (args.fy if args.fy is not None else args.fx)
        cx = args.cx if args.cx is not None else 0.0  # will be set per-frame below
        cy = args.cy if args.cy is not None else 0.0
        _K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                           dtype=torch.float32).unsqueeze(0).cuda()
        cam_int = _K
        print(f"Using fixed intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # ── Model loading ──────────────────────────────────────────────
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

    # ── Video I/O ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit frames if requested
    if args.max_frames > 0:
        total = min(total, args.max_frames)

    # Process at reduced FPS to speed up (skip frames)
    frame_step = max(1, round(fps / args.target_fps)) if args.target_fps > 0 else 1

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    out_path = os.path.join(args.output_dir, f"{video_name}_skeleton.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / frame_step
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

    print(f"\nVideo: {width}x{height} @ {fps:.1f}fps  |  {total} frames to process")
    print(f"Frame step: {frame_step} (processing every {frame_step} frame(s))")
    print(f"Output: {out_path}\n")

    # ── Per-frame processing ───────────────────────────────────────
    frame_idx = 0
    processed = 0
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

        # Convert BGR→RGB for the estimator
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Update principal point if using fixed intrinsics (cx/cy may be zero if not set)
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
            frame_idx += 1
            processed += 1
            continue

        inf_t = time.time() - t0
        inference_times.append(inf_t)

        # Save PLY meshes
        if args.save_ply:
            ply_paths = save_ply_for_frame(outputs, estimator.faces, ply_dir, frame_idx)

        # Draw and write
        vis_frame = draw_results_on_frame(frame_bgr, outputs, visualizer)
        writer.write(vis_frame)

        processed += 1
        avg_fps = 1.0 / (sum(inference_times) / len(inference_times))
        print(f"  [{processed}/{total//frame_step}] frame {frame_idx:5d} | "
              f"{inf_t:.2f}s | {len(outputs)} person(s) | avg {avg_fps:.2f} fps",
              flush=True)

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nDone! {processed} frames processed.")
    if inference_times:
        print(f"Avg inference: {sum(inference_times)/len(inference_times):.2f}s/frame "
              f"({1/(sum(inference_times)/len(inference_times)):.2f} fps)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast SAM 3D Body Video Demo")
    parser.add_argument("--video_path", default="./videos/aitor_garden_walk.mp4")
    parser.add_argument("--output_dir", default="./output_video")
    parser.add_argument("--detector", default="yolo_pose")
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.pt")
    parser.add_argument("--hand_box_source", default="yolo_pose")
    parser.add_argument("--local_checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    parser.add_argument("--target_fps", type=float, default=0,
                        help="Process at this FPS (0=all frames). E.g. 5 = every 6th frame at 30fps.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many input frames (0=all).")
    parser.add_argument("--save_ply", action="store_true",
                        help="Save a PLY mesh file per person per frame.")
    parser.add_argument("--inference_type", default="full", choices=["full", "body"],
                        help="'body' skips hand decoder for faster inference.")
    parser.add_argument("--fx", type=float, default=None,
                        help="Camera focal length x (pixels). Skips MoGe FOV estimation if set.")
    parser.add_argument("--fy", type=float, default=None,
                        help="Camera focal length y (pixels). Defaults to --fx if not set.")
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point x (pixels). Defaults to frame_width/2.")
    parser.add_argument("--cy", type=float, default=None,
                        help="Principal point y (pixels). Defaults to frame_height/2.")
    args = parser.parse_args()
    main(args)
