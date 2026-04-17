#!/usr/bin/env python3
"""Pick anatomical vertex indices on the MHR 18439-vertex skinned mesh.

MHR mesh topology is deterministic, so vertex idx X always corresponds to the
same anatomical location across subjects/frames — pick once, reuse forever.

Pipeline:
  1. Run MHR inference on a single frame (ideally a T-pose facing camera).
  2. For each target anatomical landmark (knee / ankle medial & lateral,
     wrist styloids, ASIS / PSIS / greater trochanter), derive a local
     anatomical direction from the 70 keypoints, filter mesh vertices in
     an ROI sphere around the joint center, and pick the extremum along
     the direction.
  3. Write a CSV of all vertices, a JSON of picked indices, and an
     interactive plotly HTML with joints (blue) + keypoints (red) +
     picked candidates (green, labelled) on top of the full mesh.

Reference-pose requirements:
  • Person directly facing the camera (front view)
  • Legs straight, feet roughly parallel, knees not bent
  • Arms in anatomical position (palms forward, thumbs laterally out)
  • No occlusion at the target landmarks

Usage:
    conda activate fast_sam_3d_body
    python tools/pick_anatomical_vertices.py --video_path videos/Tpose.mp4 --fx 1371

Outputs (in --out_dir, default ./tools):
    mhr_vertices.csv            all 18439 vertices  (idx, x, y, z)
    anatomical_landmarks.json   picked {name: vertex_idx}
    anatomical_landmarks.html   interactive plot — hover to inspect, click
                                legend entries to toggle traces
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


# ── Landmark definitions ───────────────────────────────────────────────────────
# Each entry:
#   center : (src, idx)          where src is "kpt" (70) or "jc" (127)
#   direction : callable(ctx) -> unit 3-vec, where ctx holds axes / kpts
#   radius : float — ROI sphere radius around the center (metres)
#
# Directions are computed from the inference frame so results are robust to
# subject orientation (we don't assume world-X = lateral).

def _axes(kpts, jc):
    """Build orthonormal anatomical axes from the 70 keypoints.

    Axis construction order (robust to a tilted MHR output mesh — the raw
    inference frame can be several degrees off vertical):

      1. sup   = normal of the best-fit plane through 6 foot keypoints
                 (2 heels + 4 toes). This is the true gravity-vertical when
                 the subject is standing on a flat floor. Orientation
                 enforced via head keypoint (Nose).
      2. lat_R = (RHip - LHip), then reprojected perpendicular to sup so
                 it's a pure medio-lateral axis (no residual vertical).
      3. ant   = cross(lat_R, sup), right-handed (lat_R → right, sup → up,
                 ant → front).

    Feet used (MHR70): LBigToe=15, LSmallToe=16, LHeel=17,
                       RBigToe=18, RSmallToe=19, RHeel=20.
    """
    foot_pts = np.stack([kpts[15], kpts[16], kpts[17],
                         kpts[18], kpts[19], kpts[20]])
    foot_center = foot_pts.mean(axis=0)
    centered = foot_pts - foot_center
    # SVD of centered points: last right-singular vector = plane normal
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    sup = vh[-1]
    # Enforce upward orientation using Nose (kpt 0) which is well above feet
    if np.dot(sup, kpts[0] - foot_center) < 0:
        sup = -sup
    sup = sup / (np.linalg.norm(sup) + 1e-9)

    # Lateral axis: LHip=9, RHip=10. Strip any vertical component so it
    # lies in the true transverse plane.
    lat_R = kpts[10] - kpts[9]
    lat_R = lat_R - np.dot(lat_R, sup) * sup
    lat_R = lat_R / (np.linalg.norm(lat_R) + 1e-9)
    lat_L = -lat_R

    # Anterior axis: cross(lat_R, sup) — right-handed.
    ant = np.cross(lat_R, sup)
    ant = ant / (np.linalg.norm(ant) + 1e-9)

    # Thumb → radial direction, projected onto the plane perpendicular to the
    # forearm axis. This isolates the true medio-lateral component so the pick
    # works in any standing pose (arms hanging, A-pose, T-pose) — otherwise
    # the thumb-base sits distally below the wrist and the raw direction is
    # dominated by the inferior axis.
    # RElbow=8, RWrist=41, RThumb base=21. LElbow=7, LWrist=62, LThumb base=42.
    def _project_perp(v, axis):
        return v - (v @ axis) * axis
    fore_R = kpts[41] - kpts[8]
    fore_R = fore_R / (np.linalg.norm(fore_R) + 1e-9)
    rad_R = _project_perp(kpts[21] - kpts[41], fore_R)
    rad_R = rad_R / (np.linalg.norm(rad_R) + 1e-9)
    fore_L = kpts[62] - kpts[7]
    fore_L = fore_L / (np.linalg.norm(fore_L) + 1e-9)
    rad_L = _project_perp(kpts[42] - kpts[62], fore_L)
    rad_L = rad_L / (np.linalg.norm(rad_L) + 1e-9)

    return {
        "lat_R": lat_R, "lat_L": lat_L,
        "sup": sup, "inf": -sup,
        "ant": ant, "post": -ant,
        "rad_R": rad_R, "uln_R": -rad_R,
        "rad_L": rad_L, "uln_L": -rad_L,
    }


def _landmarks():
    """Return the list of anatomical landmarks to pick.

    Dict per landmark. dir is a single axis key or [(axis, weight), ...] combo.
    Optional slab: constrain candidates to a slab of ±slab_half along slab_axis
    around the ROI center — useful to exclude unrelated body parts that happen
    to fall inside the sphere (e.g. a hand hanging near the trochanter).
    """
    return [
        # Knee: medial & lateral femoral epicondyles — SI slab ±3cm to keep
        # picks at joint-line height (otherwise the widest lateral vertex
        # within the ROI is pulled down toward the fibular head / upper tibia).
        {"name": "RKneeLat",  "center": ("kpt", 12), "dir": "lat_R", "radius": 0.08,
         "slabs": [("sup", 0.015)]},
        {"name": "RKneeMed",  "center": ("kpt", 12), "dir": "lat_L", "radius": 0.08,
         "slabs": [("sup", 0.015)]},
        {"name": "LKneeLat",  "center": ("kpt", 11), "dir": "lat_L", "radius": 0.08,
         "slabs": [("sup", 0.015)]},
        {"name": "LKneeMed",  "center": ("kpt", 11), "dir": "lat_R", "radius": 0.08,
         "slabs": [("sup", 0.015)]},
        # Ankle: lateral (fibular) & medial (tibial) malleoli — loose SI slab
        # ±2.5cm; medial malleolus is anatomically ~1-2cm higher than lateral.
        {"name": "RAnkleLat", "center": ("kpt", 14), "dir": "lat_R", "radius": 0.06,
         "slabs": [("sup", 0.025)]},
        {"name": "RAnkleMed", "center": ("kpt", 14), "dir": "lat_L", "radius": 0.06,
         "slabs": [("sup", 0.025)]},
        {"name": "LAnkleLat", "center": ("kpt", 13), "dir": "lat_L", "radius": 0.06,
         "slabs": [("sup", 0.025)]},
        {"name": "LAnkleMed", "center": ("kpt", 13), "dir": "lat_R", "radius": 0.06,
         "slabs": [("sup", 0.025)]},
        # Wrist: radial (thumb-side) & ulnar (pinky-side) styloids
        {"name": "RWristRad", "center": ("kpt", 41), "dir": "rad_R", "radius": 0.05},
        {"name": "RWristUln", "center": ("kpt", 41), "dir": "uln_R", "radius": 0.05},
        {"name": "LWristRad", "center": ("kpt", 62), "dir": "rad_L", "radius": 0.05},
        {"name": "LWristUln", "center": ("kpt", 62), "dir": "uln_L", "radius": 0.05},
        # Pelvis: ASIS (anterior-superior) & PSIS (posterior-superior)
        {"name": "RASIS", "center": ("kpt", 10), "dir": [("ant", 1.0), ("sup", 0.4)], "radius": 0.12},
        {"name": "LASIS", "center": ("kpt",  9), "dir": [("ant", 1.0), ("sup", 0.4)], "radius": 0.12},
        {"name": "RPSIS", "center": ("kpt", 10), "dir": [("post", 1.0), ("sup", 0.4)], "radius": 0.12},
        {"name": "LPSIS", "center": ("kpt",  9), "dir": [("post", 1.0), ("sup", 0.4)], "radius": 0.12},
        # Greater trochanter: most lateral point at hip level. Large ROI +
        # dual slab (SI ±5cm + AP ±5cm around hip joint) to exclude a relaxed
        # arm hanging at hip height (the arm sits anterior to the hip, so the
        # AP constraint removes it while leaving the posterolateral femur).
        {"name": "RTroch", "center": ("kpt", 10), "dir": "lat_R", "radius": 0.18,
         "slabs": [("sup", 0.05), ("ant", 0.05)]},
        {"name": "LTroch", "center": ("kpt",  9), "dir": "lat_L", "radius": 0.18,
         "slabs": [("sup", 0.05), ("ant", 0.05)]},
    ]


def _resolve_dir(spec, axes):
    if isinstance(spec, str):
        return axes[spec]
    v = np.zeros(3)
    for key, w in spec:
        v = v + w * axes[key]
    return v / (np.linalg.norm(v) + 1e-9)


def _resolve_center(src, idx, kpts, jc):
    return kpts[idx] if src == "kpt" else jc[idx]


def _pick_vertex(verts, center, direction, radius, slabs=None):
    """slabs: list of (axis_unit_vec, half_thickness_m) pairs — AND-combined."""
    rel = verts - center[None, :]
    dist = np.linalg.norm(rel, axis=1)
    mask = dist < radius
    if slabs:
        for axis, half in slabs:
            mask = mask & (np.abs(rel @ axis) < half)
    if not mask.any():
        return None, None
    cand_idx = np.where(mask)[0]
    dots = rel[cand_idx] @ direction
    winner = int(cand_idx[int(np.argmax(dots))])
    return winner, cand_idx.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--frame", type=int, default=0, help="Start frame (0-based)")
    parser.add_argument("--avg_frames", type=int, default=1,
                        help="Average verts/kpts/jc over N consecutive frames starting at --frame (more stable)")
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
    # Skip to start frame
    for _ in range(args.frame):
        cap.read()

    acc_verts = None
    acc_kpts  = None
    acc_jc    = None
    n_ok = 0
    h = w = None
    for i in range(args.avg_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        cam_int = None
        if args.fx:
            cam_int = torch.tensor([[args.fx, 0, w/2], [0, args.fx, h/2], [0, 0, 1]],
                                   dtype=torch.float32).unsqueeze(0).cuda()
        outputs = estimator.process_one_image(
            frame_rgb, hand_box_source="yolo_pose", inference_type="body",
            cam_int=cam_int,
        )
        if not outputs:
            print(f"  frame {args.frame + i}: no person detected, skipping")
            continue
        p = outputs[0]
        v = p.get("pred_vertices")
        k = p.get("pred_keypoints_3d")
        j = p.get("pred_joint_coords")
        ct = p.get("pred_cam_t")
        if v is None or np.any(np.isnan(v)):
            continue
        # Put in world space so averaging is anchored consistently
        v_w = v + ct[None, :]
        k_w = k + ct[None, :]
        j_w = j + ct[None, :]
        if acc_verts is None:
            acc_verts = v_w.astype(np.float64)
            acc_kpts  = k_w.astype(np.float64)
            acc_jc    = j_w.astype(np.float64)
        else:
            acc_verts += v_w
            acc_kpts  += k_w
            acc_jc    += j_w
        n_ok += 1
    cap.release()
    if n_ok == 0:
        print("No usable frames. Aborting.")
        return
    verts_w = (acc_verts / n_ok).astype(np.float32)
    kpts_w  = (acc_kpts  / n_ok).astype(np.float32)
    jc_w    = (acc_jc    / n_ok).astype(np.float32)
    print(f"Averaged {n_ok}/{args.avg_frames} frames starting at frame {args.frame} ({w}x{h})")
    print(f"Got {verts_w.shape[0]} vertices, {kpts_w.shape[0]} kpts, {jc_w.shape[0]} joints")

    # Y-up visualisation flip, same convention as dump_mhr_joints.py
    def flip(a):
        b = a.copy()
        b[:, 1] = -b[:, 1]
        b[:, 0] = -b[:, 0]
        return b
    vv, kv, jv = flip(verts_w), flip(kpts_w), flip(jc_w)

    # ── Build anatomical axes from the 70 keypoints (post-flip) ──────────────
    axes = _axes(kv, jv)
    print("Anatomical axes (metres, Y-up frame):")
    for k, v in axes.items():
        print(f"  {k}: [{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]")

    # ── Pick each landmark ──────────────────────────────────────────────────
    picks = {}
    cand_per_lm = {}
    for lm in _landmarks():
        name = lm["name"]
        src, idx = lm["center"]
        radius = lm["radius"]
        center = _resolve_center(src, idx, kv, jv)
        direction = _resolve_dir(lm["dir"], axes)
        slabs = [(axes[ax], h) for ax, h in lm["slabs"]] if "slabs" in lm else None
        winner, cand = _pick_vertex(vv, center, direction, radius, slabs)
        if winner is None:
            print(f"  {name:10s} NO candidates in r={radius*100:.1f}cm sphere")
            continue
        picks[name] = winner
        cand_per_lm[name] = cand
        wp = vv[winner]
        slab_info = ""
        if "slabs" in lm:
            slab_info = ", slabs=" + "+".join(f"{ax}±{h*100:.1f}cm" for ax, h in lm["slabs"])
        print(f"  {name:10s} v{winner:5d}  ({wp[0]:+.3f} {wp[1]:+.3f} {wp[2]:+.3f})  "
              f"(center={src}{idx}, r={radius*100:.1f}cm{slab_info}, N_cand={len(cand)})")

    # ── Write vertices CSV ──────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "mhr_vertices.csv"
    with csv_path.open("w") as f:
        f.write("idx,x,y,z\n")
        for i in range(vv.shape[0]):
            f.write(f"{i},{vv[i,0]:.6f},{vv[i,1]:.6f},{vv[i,2]:.6f}\n")
    print(f"Wrote {csv_path}")

    # ── Write landmark JSON ─────────────────────────────────────────────────
    json_path = out_dir / "anatomical_landmarks.json"
    json_path.write_text(json.dumps({
        "mhr_mesh_topology": "deterministic_18439_verts",
        "reference_video": args.video_path,
        "reference_frame": args.frame,
        "picked_vertex_indices": picks,
    }, indent=2))
    print(f"Wrote {json_path}")

    # ── Build interactive HTML ───────────────────────────────────────────────
    def trace_scatter(points, text, name, size, color, opacity=1.0):
        return {
            "x": points[:, 0].tolist(),
            "y": points[:, 2].tolist(),   # plotly z → Y (depth)
            "z": points[:, 1].tolist(),   # plotly "up" = our Y
            "text": text,
            "name": name,
            "mode": "markers",
            "type": "scatter3d",
            "marker": {"size": size, "color": color, "opacity": opacity},
        }

    # Full mesh cloud (dense → semi-transparent, small dots)
    mesh_trace = trace_scatter(
        vv, [f"v{i}" for i in range(vv.shape[0])],
        "mesh verts (18439)", size=1.5, color="lightgray", opacity=0.25,
    )
    joint_trace = trace_scatter(
        jv, [f"j{i}" for i in range(jv.shape[0])],
        "joints (127)", size=4, color="blue", opacity=0.9,
    )
    kpt_trace = trace_scatter(
        kv, [f"kp{i}" for i in range(kv.shape[0])],
        "keypoints (70)", size=3, color="red", opacity=0.8,
    )
    # Picked landmarks trace
    if picks:
        pick_pts = np.stack([vv[vi] for vi in picks.values()])
        pick_txt = [f"{name} (v{vi})" for name, vi in picks.items()]
        pick_trace = trace_scatter(
            pick_pts, pick_txt, "picked landmarks", size=6, color="green",
        )
        pick_trace["mode"] = "markers+text"
        pick_trace["textposition"] = "top center"
        pick_trace["textfont"] = {"size": 10, "color": "darkgreen"}
    else:
        pick_trace = None

    # Per-landmark ROI traces (all candidate verts within the sphere)
    roi_traces = []
    palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, (name, cand) in enumerate(cand_per_lm.items()):
        if not cand:
            continue
        pts = vv[cand]
        txt = [f"v{ci} ({name})" for ci in cand]
        t = trace_scatter(pts, txt, f"ROI:{name}", size=2.5,
                          color=palette[i % len(palette)], opacity=0.7)
        t["visible"] = "legendonly"  # toggle-on in UI
        roi_traces.append(t)

    traces = [mesh_trace, joint_trace, kpt_trace]
    if pick_trace:
        traces.append(pick_trace)
    traces.extend(roi_traces)

    plotly_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<script src="{plotly_cdn}"></script>
</head>
<body>
<h2>MHR anatomical vertex picker</h2>
<p>Hover for idx. Green = auto-picked landmarks.  ROI:* traces start hidden
— click the legend to show candidate verts around each joint.</p>
<div id="plot" style="width:100%;height:90vh;"></div>
<script>
var traces = {json.dumps(traces)};
Plotly.newPlot('plot', traces, {{scene: {{aspectmode: 'data',
      xaxis: {{title: 'X (lateral)'}},
      yaxis: {{title: 'Z (anterior)'}},
      zaxis: {{title: 'Y (up)'}}}},
    title: 'MHR mesh ({vv.shape[0]} verts) + joints + kpts + picked landmarks'}});
</script></body></html>"""

    html_path = out_dir / "anatomical_landmarks.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"\nOpen {html_path} in a browser to confirm the picks.")
    print("Edit anatomical_landmarks.json to override any wrong idx by hand.")


if __name__ == "__main__":
    main()
