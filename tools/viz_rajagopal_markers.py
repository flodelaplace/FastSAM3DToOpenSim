#!/usr/bin/env python3
"""Visualize Rajagopal mocap markers on the MHR mesh for validation.

For each marker in assets/rajagopal2015_mocap.osim, plots its MHR source:
  - GREEN  = direct MHR 70-kpt or 127-joint armature index
  - ORANGE = vertex pick on the skinned mesh (sphere-ROI heuristics:
             lateral/anterior directions derived from feet + hip + nose,
             optional slab constraints and center offsets per landmark)

Labels appear above each marker (Rajagopal name). Mesh/joints/kpts are shown
as background layers and can be toggled via the legend.

Typical workflow:
    1. Pick a reference video with the subject facing the camera, standing
       with arms relaxed by the sides.
    2. Run this tool (averaging 10 frames dampens single-frame kpt noise).
    3. Open the HTML alongside the .osim in OpenSim GUI; for each marker,
       check the orange dot lands on the intended anatomical landmark.
    4. If a vertex pick is off, edit the landmark spec in the SOURCES dict
       (adjust radius / slabs / direction bias / center_offset), rerun.
    5. Once happy, freeze the picks into
       assets/rajagopal_anatomical_vertex_idx.json (used by the runtime
       converter in sam_3d_body/export/rajagopal_converter.py).

Usage:
    conda activate fast_sam_3d_body
    python tools/viz_rajagopal_markers.py \\
        --video_path videos/Squat.MP4 \\
        --avg_frames 10

Key args:
    --video_path PATH    (required) input video
    --frame N            (default 0) starting frame
    --avg_frames N       (default 1) average N consecutive frames — use
                         >=10 for stability; longer helps with noisy kpts
    --fx FLOAT           focal length (pixels); if omitted, FOV is estimated
                         from MoGe2 per frame
    --detector_model P   YOLO-Pose TensorRT engine (default ./checkpoints/...)
    --out_dir DIR        where to write outputs (default ./tools)

Outputs (in --out_dir):
    rajagopal_markers_viz.html   interactive plotly 3D view
    rajagopal_markers_viz.json   {marker_name: {pos:[x,y,z], source:"kpt70|jcoord127|vertex vNNNN"}}

The JSON is what you extract vertex indices from when freezing picks.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


# ── Axes (foot-plane vertical + hip lateral; identical to picker) ─────────────
def _axes(kpts):
    foot_pts = np.stack([kpts[15], kpts[16], kpts[17],
                         kpts[18], kpts[19], kpts[20]])
    centered = foot_pts - foot_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    sup = vh[-1]
    if np.dot(sup, kpts[0] - foot_pts.mean(axis=0)) < 0:
        sup = -sup
    sup = sup / (np.linalg.norm(sup) + 1e-9)

    lat_R = kpts[10] - kpts[9]
    lat_R = lat_R - np.dot(lat_R, sup) * sup
    lat_R = lat_R / (np.linalg.norm(lat_R) + 1e-9)
    ant = np.cross(lat_R, sup)
    ant = ant / (np.linalg.norm(ant) + 1e-9)
    # Enforce anterior orientation: the nose must be anterior of the shoulder
    # midpoint, otherwise cross(lat_R, sup) gave us the posterior direction
    # (happens when the subject faces the camera — X flip inverts the sign).
    nose_dir = kpts[0] - 0.5 * (kpts[5] + kpts[6])
    if np.dot(ant, nose_dir) < 0:
        ant = -ant

    def _perp(v, axis):
        return v - (v @ axis) * axis
    fore_R = kpts[41] - kpts[8]
    fore_R = fore_R / (np.linalg.norm(fore_R) + 1e-9)
    rad_R = _perp(kpts[21] - kpts[41], fore_R)
    rad_R = rad_R / (np.linalg.norm(rad_R) + 1e-9)
    fore_L = kpts[62] - kpts[7]
    fore_L = fore_L / (np.linalg.norm(fore_L) + 1e-9)
    rad_L = _perp(kpts[42] - kpts[62], fore_L)
    rad_L = rad_L / (np.linalg.norm(rad_L) + 1e-9)

    return {"sup": sup, "inf": -sup,
            "lat_R": lat_R, "lat_L": -lat_R,
            "ant": ant, "post": -ant,
            "rad_R": rad_R, "uln_R": -rad_R,
            "rad_L": rad_L, "uln_L": -rad_L}


def _resolve_dir(spec, axes):
    if isinstance(spec, str):
        return axes[spec]
    v = np.zeros(3)
    for key, w in spec:
        v = v + w * axes[key]
    return v / (np.linalg.norm(v) + 1e-9)


def _pick_vertex(verts, center, direction, radius, slabs, center_offset=None):
    """
    center_offset: optional list of (axis_unit_vec, offset_m) — shifts the
    effective ROI center by sum_i(offset_i * axis_i). Useful to bias the
    search to one side of a kpt (e.g. 1cm above wrist for bone styloids).
    """
    ctr = center.copy()
    if center_offset:
        for ax, off in center_offset:
            ctr = ctr + off * ax
    rel = verts - ctr[None, :]
    dist = np.linalg.norm(rel, axis=1)
    mask = dist < radius
    if slabs:
        for ax, h in slabs:
            mask = mask & (np.abs(rel @ ax) < h)
    if not mask.any():
        return None
    cand_idx = np.where(mask)[0]
    dots = rel[cand_idx] @ direction
    return int(cand_idx[int(np.argmax(dots))])


# ── Rajagopal marker → MHR source ────────────────────────────────────────────
# ("kpt70", idx)       — directly from pred_keypoints_3d[idx]
# ("jcoord127", idx)   — directly from pred_joint_coords[idx]
# ("vertex", spec)     — vertex pick; spec = (center_src, center_idx,
#                        dir_key_or_combo, radius_m, slabs_or_None)
#
# Hand note: current keypoint_converter.py names kpts 25/37/46/58 as finger
# "tips". If that's correct, RIndex (Coco133 MCP position) would need a
# different kpt than RIndexTip, but we don't have a named MCP kpt. Shown here
# mapped to the same kpt — to flag the potential mismatch visually.
SOURCES = {
    # Head
    "Nose": ("kpt70", 0), "LEye": ("kpt70", 1), "REye": ("kpt70", 2),
    "LEar": ("kpt70", 3), "REar": ("kpt70", 4),
    "HTOP": ("jcoord127", 126),
    # Spine
    "c_spine0": ("jcoord127", 34),
    "c_spine1": ("jcoord127", 35),
    "c_spine2": ("jcoord127", 36),
    "c_spine3": ("jcoord127", 37),
    "c_neck":   ("jcoord127", 110),
    "c_head":   ("jcoord127", 113),
    "RCLAV":    ("jcoord127", 74),
    "LCLAV":    ("jcoord127", 38),
    # Joint centers (SJC dropped — unreliable; see FlodelaplaceConverter)
    "LEJC": ("kpt70", 7),  "REJC": ("kpt70", 8),
    "LHJC": ("kpt70", 9),  "RHJC": ("kpt70", 10),
    "LKJC": ("kpt70", 11), "RKJC": ("kpt70", 12),
    "LAJC": ("kpt70", 13), "RAJC": ("kpt70", 14),
    # Acromions
    "LACR": ("kpt70", 67), "RACR": ("kpt70", 68),
    # Foot
    "LTOE": ("kpt70", 15), "LMT5": ("kpt70", 16), "LCAL": ("kpt70", 17),
    "RTOE": ("kpt70", 18), "RMT5": ("kpt70", 19), "RCAL": ("kpt70", 20),
    # Wrists
    "RWrist_hand": ("kpt70", 41), "LWrist_hand": ("kpt70", 62),
    # Hand: metacarpals (kpt-X+3 = most proximal joint) + tips (kpt-X).
    # Verified on Squat.MP4: thumb 21=tip/24=MCP, index 25=tip/28=MCP,
    # pinky 37=tip/40=MCP; same pattern for the L mirror (42+3/46+3/58+3).
    "RThumb":    ("kpt70", 24),
    "RIndex":    ("kpt70", 28),
    "RPinky":    ("kpt70", 40),
    "RIndexTip": ("kpt70", 25),
    "RPinkyTip": ("kpt70", 37),
    "LThumb":    ("kpt70", 45),
    "LIndex":    ("kpt70", 49),
    "LPinky":    ("kpt70", 61),
    "LIndexTip": ("kpt70", 46),
    "LPinkyTip": ("kpt70", 58),
    # Vertex picks — femoral condyles.
    # Dual slab (sup ±12mm + ant ±30mm) keeps picks near the joint line both
    # vertically and in AP — without the ant slab, the lateral extremum
    # sometimes drifted +6cm anterior (to the front-thigh skin).
    # All 4 knee condyles shifted slightly anterior (+8mm ant) — user feedback
    # indicated they sat a touch behind the actual epicondyles.
    "RLFC": ("vertex", ("kpt", 12, "lat_R", 0.08, [("sup", 0.012), ("ant", 0.030)], [("sup", 0.008), ("ant", 0.008)])),
    "LLFC": ("vertex", ("kpt", 11, "lat_L", 0.08, [("sup", 0.012), ("ant", 0.030)], [("sup", 0.008), ("ant", 0.008)])),
    # Medial condyles sit slightly inferior to the joint line anatomically —
    # shift center -3mm sup (= 3mm down) vs the laterals, same ant nudge.
    "RMFC": ("vertex", ("kpt", 12, "lat_L", 0.08, [("sup", 0.012), ("ant", 0.030)], [("sup", -0.003), ("ant", 0.008)])),
    "LMFC": ("vertex", ("kpt", 11, "lat_R", 0.08, [("sup", 0.012), ("ant", 0.030)], [("sup", -0.003), ("ant", 0.008)])),
    # Malleoli
    # Lateral malleolus: sup bias + tighter slab — pure-lateral max on its own
    # landed ~3cm BELOW the ankle joint center (probably on the lower fibula
    # bulge or calcaneus). Bias toward upper-lateral brings the pick up to
    # joint-line / distal-fibula level where the malleolus protrudes.
    "RLMAL": ("vertex", ("kpt", 14, [("lat_R", 1.0), ("sup", 0.4)], 0.06, [("sup", 0.015)])),
    "LLMAL": ("vertex", ("kpt", 13, [("lat_L", 1.0), ("sup", 0.4)], 0.06, [("sup", 0.015)])),
    # Medial malleolus: user reported OK, keep pure-lateral direction.
    "RMMAL": ("vertex", ("kpt", 14, "lat_L", 0.06, [("sup", 0.025)])),
    "LMMAL": ("vertex", ("kpt", 13, "lat_R", 0.06, [("sup", 0.025)])),
    # ASIS / PSIS — center_offset + pure direction (more predictable than
    # weighted-direction because post/lat/ant axes are entangled in world
    # space when the mesh is tilted).
    #   ASIS: shift ROI center +5cm sup + 2cm post from HJC, then pick the
    #   most-anterior surface vertex within 8cm. Anatomically ASIS sits
    #   ~5-8cm superior + ~3-5cm anterior of HJC on the iliac crest.
    "RASI": ("vertex", ("kpt", 10, "ant", 0.12, None, [("sup", 0.09), ("post", 0.02)])),
    "LASI": ("vertex", ("kpt",  9, "ant", 0.12, None, [("sup", 0.09), ("post", 0.02)])),
    #   PSIS: shift ROI center medial + superior from HJC, then pick the
    #   most-posterior vertex. Superior offset raises them above HJC (feedback
    #   showed the previous LPSI=v6209 / RPSI=v7326 were 5-6cm too low in IK).
    # RPSI/LPSI frozen — raised +6.6cm sup vs user's original picks (v7326,
    # v6209). Iteration history: orig → +5cm (v7442/v6327, still too low) →
    # +6.6cm (v7441/v6328, current).
    "RPSI": ("v", 7319),
    "LPSI": ("v", 6216),
    # Forearm distal = wrist styloids. Shift ROI center +10mm sup → picks
    # constrained between -0.5cm and +2.5cm relative to the wrist kpt, i.e.
    # the bone styloids just proximal of the wrist joint.
    "RFAradius": ("vertex", ("kpt", 41, [("rad_R", 1.0), ("sup", 0.2)], 0.05, [("sup", 0.010)], [("sup", 0.010)])),
    "RFAulna":   ("vertex", ("kpt", 41, [("uln_R", 1.0), ("sup", 0.2)], 0.05, [("sup", 0.010)], [("sup", 0.010)])),
    "LFAradius": ("vertex", ("kpt", 62, [("rad_L", 1.0), ("sup", 0.2)], 0.05, [("sup", 0.010)], [("sup", 0.010)])),
    "LFAulna":   ("vertex", ("kpt", 62, [("uln_L", 1.0), ("sup", 0.2)], 0.05, [("sup", 0.010)], [("sup", 0.010)])),
    # Humeral epicondyles
    # Lateral: sup bias + tighter slab — was ~2.5cm below elbow joint center.
    "RLEL": ("vertex", ("kpt", 8, [("lat_R", 1.0), ("sup", 0.4)], 0.06, [("sup", 0.015)])),
    "LLEL": ("vertex", ("kpt", 7, [("lat_L", 1.0), ("sup", 0.4)], 0.06, [("sup", 0.015)])),
    # Medial: shift search slightly down (-3mm) so picks land a touch below
    # the elbow joint center, matching the anatomical medial epicondyle.
    "RMEL": ("vertex", ("kpt", 8, "lat_L", 0.06, [("sup", 0.015)], [("sup", -0.003)])),
    "LMEL": ("vertex", ("kpt", 7, "lat_R", 0.06, [("sup", 0.015)], [("sup", -0.003)])),
    # C7 — posterior skin at c_neck level. c_neck jcoord is internal (on the
    # cervical vertebra), so the ROI needs to reach through the muscle/skin
    # on the posterior side. Center shifted +3cm sup to raise the pick —
    # previous pick was ~4cm too low vs the .osim template position.
    # C7 : ajout d'un slab lat_R ±1.5cm pour forcer le pick sur la ligne
    # médiane. Sans ça, le c_neck (jcoord 110) lui-même peut être un peu
    # latéralisé et le vertex post-c_neck hérite du décalage (~30 mm
    # systématique observé sur 5 sujets).
    "C7":   ("vertex", ("jc", 110, "post", 0.10, [("sup", 0.025), ("lat_R", 0.015)], [("sup", 0.05)])),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--avg_frames", type=int, default=1)
    parser.add_argument("--out_dir", default="./tools")
    args = parser.parse_args()

    import cv2, torch
    from notebook.utils import setup_sam_3d_body

    print("Loading model...")
    estimator = setup_sam_3d_body(
        detector_name="yolo_pose", detector_model=args.detector_model,
        local_checkpoint_path="./checkpoints/sam-3d-body-dinov3",
    )

    cap = cv2.VideoCapture(args.video_path)
    for _ in range(args.frame):
        cap.read()
    acc_v = acc_k = acc_j = None
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
            continue
        p = outputs[0]
        v = p.get("pred_vertices"); k = p.get("pred_keypoints_3d")
        j = p.get("pred_joint_coords"); ct = p.get("pred_cam_t")
        if v is None or np.any(np.isnan(v)):
            continue
        vw = v + ct[None, :]; kw = k + ct[None, :]; jw = j + ct[None, :]
        if acc_v is None:
            acc_v, acc_k, acc_j = vw.astype(np.float64), kw.astype(np.float64), jw.astype(np.float64)
        else:
            acc_v += vw; acc_k += kw; acc_j += jw
        n_ok += 1
    cap.release()
    if n_ok == 0:
        print("No usable frames."); return
    verts_w = (acc_v / n_ok).astype(np.float32)
    kpts_w  = (acc_k / n_ok).astype(np.float32)
    jc_w    = (acc_j / n_ok).astype(np.float32)
    print(f"Averaged {n_ok}/{args.avg_frames} frames ({w}x{h})")

    # Y-up flip (visualization-only)
    def flip(a):
        b = a.copy(); b[:, 1] = -b[:, 1]; b[:, 0] = -b[:, 0]; return b
    vv, kv, jv = flip(verts_w), flip(kpts_w), flip(jc_w)

    axes = _axes(kv)
    nose_shoulder = kv[0] - 0.5 * (kv[5] + kv[6])
    print(f"[axes] lat_R = {axes['lat_R']}")
    print(f"[axes] sup   = {axes['sup']}")
    print(f"[axes] ant   = {axes['ant']}")
    print(f"[axes] nose-shoulder_mid = {nose_shoulder}  (should align with ant)")
    print(f"[axes] dot(ant, nose_dir) = {np.dot(axes['ant'], nose_shoulder):+.4f}  (>0 = OK)")
    clav_mid = 0.5 * (jv[74] + jv[38])
    clav_neck = clav_mid - jv[110]
    print(f"[axes] dot(ant, clav-neck) = {np.dot(axes['ant'], clav_neck):+.4f}  (>0 = OK, clavicles anterior)")

    # Resolve each marker's position
    resolved = {}  # name -> (pos[3], source_type)
    for name, (src_type, spec) in SOURCES.items():
        if src_type == "kpt70":
            resolved[name] = (kv[spec], "kpt70")
        elif src_type == "jcoord127":
            resolved[name] = (jv[spec], "jcoord127")
        elif src_type == "v":
            # Direct vertex index (user-frozen override).
            resolved[name] = (vv[spec], f"vertex v{spec}")
        elif src_type == "vertex":
            if len(spec) == 5:
                center_src, center_idx, dir_spec, radius, slabs = spec
                offset_spec = None
            else:
                center_src, center_idx, dir_spec, radius, slabs, offset_spec = spec
            center = kv[center_idx] if center_src == "kpt" else jv[center_idx]
            direction = _resolve_dir(dir_spec, axes)
            slabs_resolved = None
            if slabs:
                slabs_resolved = [(axes[ax], h) for ax, h in slabs]
            offset_resolved = None
            if offset_spec:
                offset_resolved = [(axes[ax], off) for ax, off in offset_spec]
            vi = _pick_vertex(vv, center, direction, radius, slabs_resolved, offset_resolved)
            if vi is None:
                print(f"  {name:12s} NO vertex in ROI"); continue
            resolved[name] = (vv[vi], f"vertex v{vi}")
        else:
            print(f"  {name}: unknown source {src_type}"); continue

    # Group by color: green = direct MHR, orange = vertex pick
    green = [(n, p) for n, (p, t) in resolved.items() if t in ("kpt70", "jcoord127")]
    orange = [(n, p) for n, (p, t) in resolved.items() if t.startswith("vertex")]

    # Identical trace-builder to pick_anatomical_vertices.py (plotly axes swap:
    # plot Y = world Z depth, plot Z = world Y up).
    def trace_scatter(points, text, name, size, color, opacity=1.0):
        return {
            "x": points[:, 0].tolist(),
            "y": points[:, 2].tolist(),
            "z": points[:, 1].tolist(),
            "text": text,
            "name": name,
            "mode": "markers",
            "type": "scatter3d",
            "marker": {"size": size, "color": color, "opacity": opacity},
        }

    # Background layers — same style as anatomical_landmarks.html
    mesh_trace = trace_scatter(
        vv, [f"v{i}" for i in range(vv.shape[0])],
        "mesh verts (18439)", size=1.5, color="lightgray", opacity=0.25)
    joint_trace = trace_scatter(
        jv, [f"j{i}" for i in range(jv.shape[0])],
        "joints (127)", size=4, color="blue", opacity=0.9)
    kpt_trace = trace_scatter(
        kv, [f"kp{i}" for i in range(kv.shape[0])],
        "keypoints (70)", size=3, color="red", opacity=0.8)

    # Rajagopal markers on top — labelled, bigger, always-on
    def labelled_trace(pairs, name, color):
        if not pairs:
            return None
        pts = np.stack([p for _, p in pairs])
        txt = [n for n, _ in pairs]
        t = trace_scatter(pts, txt, name, size=6, color=color, opacity=1.0)
        t["mode"] = "markers+text"
        t["textposition"] = "top center"
        t["textfont"] = {"size": 9, "color": color}
        return t

    green_trace  = labelled_trace(green,  f"MHR direct ({len(green)})", "#2ca02c")
    orange_trace = labelled_trace(orange, f"Vertex picks ({len(orange)})", "#ff7f0e")

    traces = [mesh_trace, joint_trace, kpt_trace]
    if green_trace:  traces.append(green_trace)
    if orange_trace: traces.append(orange_trace)

    plotly_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<script src="{plotly_cdn}"></script>
</head>
<body>
<h2>Rajagopal mocap markers on MHR mesh</h2>
<p><b>Mesh verts</b> gray (18439), <b>joints</b> blue (127), <b>keypoints</b> red (70).
<b>GREEN</b> = Rajagopal marker fed directly from an MHR kpt/jcoord.
<b>ORANGE</b> = Rajagopal marker fed by a vertex pick on the mesh.
Hover for the name, click any legend entry to toggle.</p>
<div id="plot" style="width:100%;height:90vh;"></div>
<script>
var traces = {json.dumps(traces)};
Plotly.newPlot('plot', traces, {{scene: {{aspectmode: 'data',
      xaxis: {{title: 'X (anterior)'}},
      yaxis: {{title: 'Z (lateral)'}},
      zaxis: {{title: 'Y (up)'}}}},
    title: 'Rajagopal markers ({len(green)} direct + {len(orange)} vertex picks) on MHR mesh'}});
</script></body></html>"""

    out_path = Path(args.out_dir) / "rajagopal_markers_viz.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"  Green (MHR direct) : {len(green)} markers")
    print(f"  Orange (vertex)    : {len(orange)} markers")
    # Dump a JSON with the picks too
    dump = {
        "reference_video": args.video_path,
        "start_frame": args.frame,
        "avg_frames": n_ok,
        "markers": {n: {"pos": p.tolist(), "source": t}
                    for n, (p, t) in resolved.items()},
    }
    json_path = Path(args.out_dir) / "rajagopal_markers_viz.json"
    json_path.write_text(json.dumps(dump, indent=2))
    print(f"  Wrote {json_path}")


if __name__ == "__main__":
    main()
