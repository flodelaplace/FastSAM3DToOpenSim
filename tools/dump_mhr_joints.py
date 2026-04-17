#!/usr/bin/env python3
"""Dump MHR 127-joint positions from a single inference frame.

Writes a CSV and a simple HTML 3D scatter plot to visually identify each joint.

Usage:
    conda activate fast_sam_3d_body
    python tools/dump_mhr_joints.py --video_path videos/Squat.mp4 --fx 1371

Outputs:
    tools/mhr_127_joints.csv   — idx, x, y, z for each joint
    tools/mhr_127_joints.html  — interactive 3D scatter plot (open in browser)
"""
import argparse
import os
import sys
import json

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--frame", type=int, default=0, help="Which frame to use (0-based)")
    parser.add_argument("--out_dir", default="./tools")
    args = parser.parse_args()

    import cv2
    import torch
    from notebook.utils import setup_sam_3d_body

    print("Loading model...")
    estimator = setup_sam_3d_body(
        detector_name="yolo_pose",
        detector_model=args.detector_model,
        local_checkpoint_path="./checkpoints/sam-3d-body-dinov3",
    )

    cap = cv2.VideoCapture(args.video_path)
    for i in range(args.frame + 1):
        ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {args.frame}")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    cam_int = None
    if args.fx:
        cam_int = torch.tensor([[args.fx, 0, w/2], [0, args.fx, h/2], [0, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).cuda()

    print(f"Running inference on frame {args.frame} ({w}x{h})...")
    outputs = estimator.process_one_image(
        frame_rgb, hand_box_source="yolo_pose", inference_type="body",
        cam_int=cam_int,
    )

    if not outputs:
        print("No person detected!")
        return

    person = outputs[0]
    jcoords = person.get("pred_joint_coords")  # (127, 3)
    kpts = person.get("pred_keypoints_3d")      # (70, 3)
    cam_t = person.get("pred_cam_t")            # (3,)

    if jcoords is None:
        print("No joint coords! Model might not output them.")
        return

    print(f"Got {jcoords.shape[0]} joint coords, {kpts.shape[0]} keypoints")

    # Move to world space (add camera translation)
    jc_world = jcoords + cam_t[None, :]
    kp_world = kpts + cam_t[None, :]

    # Flip to Y-up for visualization
    jc_vis = jc_world.copy()
    jc_vis[:, 1] = -jc_vis[:, 1]
    jc_vis[:, 0] = -jc_vis[:, 0]

    kp_vis = kp_world.copy()
    kp_vis[:, 1] = -kp_vis[:, 1]
    kp_vis[:, 0] = -kp_vis[:, 0]

    # Known joint mappings (from our existing code)
    known = {
        34: "c_spine0 (lower lumbar)",
        35: "c_spine1 (upper lumbar)",
        36: "c_spine2 (lower thoracic)",
        37: "c_spine3 (upper thoracic)",
        110: "c_neck",
        113: "c_head",
    }

    # Write CSV
    csv_path = os.path.join(args.out_dir, "mhr_127_joints.csv")
    with open(csv_path, "w") as f:
        f.write("idx,x,y,z,known_name\n")
        for i in range(jcoords.shape[0]):
            name = known.get(i, "")
            f.write(f"{i},{jc_vis[i,0]:.6f},{jc_vis[i,1]:.6f},{jc_vis[i,2]:.6f},{name}\n")
    print(f"Wrote {csv_path}")

    # Write HTML 3D scatter plot using plotly CDN
    html_path = os.path.join(args.out_dir, "mhr_127_joints.html")

    # Build data arrays for plotly
    jc_data = {
        "x": jc_vis[:, 0].tolist(),
        "y": jc_vis[:, 1].tolist(),
        "z": jc_vis[:, 2].tolist(),
        "text": [f"j{i}" + (f" ({known[i]})" if i in known else "") for i in range(127)],
    }
    kp_data = {
        "x": kp_vis[:, 0].tolist(),
        "y": kp_vis[:, 1].tolist(),
        "z": kp_vis[:, 2].tolist(),
        "text": [f"kp{i}" for i in range(70)],
    }

    html = f"""<!DOCTYPE html>
<html><head><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body>
<h2>MHR 127 Joint Coords (blue) + 70 Keypoints (red)</h2>
<p>Hover over points to see index. Known joints are labelled.</p>
<div id="plot" style="width:100%;height:90vh;"></div>
<script>
var jc = {json.dumps(jc_data)};
var kp = {json.dumps(kp_data)};
Plotly.newPlot('plot', [
  {{x: jc.x, y: jc.z, z: jc.y, text: jc.text, mode: 'markers+text',
    type: 'scatter3d', name: 'joints (127)',
    marker: {{size: 4, color: 'blue'}},
    textposition: 'top center', textfont: {{size: 8}}}},
  {{x: kp.x, y: kp.z, z: kp.y, text: kp.text, mode: 'markers',
    type: 'scatter3d', name: 'keypoints (70)',
    marker: {{size: 3, color: 'red', opacity: 0.6}}}},
], {{scene: {{aspectmode: 'data',
      xaxis: {{title: 'X (anterior)'}},
      yaxis: {{title: 'Z (lateral)'}},
      zaxis: {{title: 'Y (up)'}}}},
    title: 'MHR Armature — 127 joints + 70 keypoints'}});
</script></body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {html_path}")
    print(f"\nOpen {html_path} in your browser to explore the joints interactively.")
    print("Hover over blue dots to see joint index. Red dots are the 70 surface keypoints.")


if __name__ == "__main__":
    main()
