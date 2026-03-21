# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
OpenSim export utilities: MOT, GLB, and MarkerSet .osim from processed markers.

This module provides the low-level writers for OpenSim-format output files.
Coordinate transformation, bone normalisation, Butterworth filtering, and TRC
export are handled by the pipeline modules:

  sam_3d_body/export/post_processing.py      — PostProcessor
  sam_3d_body/export/coordinate_transform.py — CoordinateTransformer
  sam_3d_body/export/keypoint_converter.py   — KeypointConverter
  sam_3d_body/export/trc_exporter.py         — TRCExporter

Coordinate system (all inputs to these writers must already be in Y-up):
  X = forward (anterior)
  Y = up
  Z = right (lateral)
  Units: metres

Marker array layout (27 markers, matches KeypointConverter body_only output):
  0:Nose, 1:LShoulder, 2:RShoulder, 3:LElbow, 4:RElbow,
  5:LHip, 6:RHip, 7:LKnee, 8:RKnee, 9:LAnkle, 10:RAnkle,
  11:LBigToe, 12:LSmallToe, 13:LHeel, 14:RBigToe, 15:RSmallToe, 16:RHeel,
  17:RWrist, 18:LWrist, 19:LOlecranon, 20:ROlecranon,
  21:LAcromion, 22:RAcromion, 23:Neck,
  24:PelvisCenter, 25:Thorax, 26:SpineMid  (derived)
"""

from __future__ import annotations

import struct
import json
from pathlib import Path
from typing import List

import numpy as np

# ── Marker names (order matches KeypointConverter body_only + derived output) ─
# Indices 0-4: head   5-6: shoulder   7-8: elbow   9-10: hip  11-12: knee
# 13-14: ankle  15-20: feet  21-22: wrist  23-24: olecranon
# 25-26: cubital fossa  27-28: acromion  29: neck  30-32: derived
BODY_MARKER_NAMES = [
    "Nose",
    "LEye",      "REye",
    "LEar",      "REar",
    "LShoulder", "RShoulder",
    "LElbow",    "RElbow",
    "LHip",      "RHip",
    "LKnee",     "RKnee",
    "LAnkle",    "RAnkle",
    "LBigToe",   "LSmallToe", "LHeel",
    "RBigToe",   "RSmallToe", "RHeel",
    "RWrist",    "LWrist",
    "LOlecranon", "ROlecranon",
    "LCubitalFossa", "RCubitalFossa",
    "LAcromion",  "RAcromion",
    "Neck",
    # Derived markers (appended by KeypointConverter)
    "PelvisCenter", "Thorax", "SpineMid",
]

# Skeleton connectivity for GLB line visualisation (marker index pairs)
# Indices reflect the new BODY_MARKER_NAMES ordering above.
SKELETON_LINKS = [
    (5,  6),  # LShoulder  - RShoulder
    (5,  7),  # LShoulder  - LElbow
    (7,  22), # LElbow     - LWrist
    (6,  8),  # RShoulder  - RElbow
    (8,  21), # RElbow     - RWrist
    (5,  9),  # LShoulder  - LHip
    (6,  10), # RShoulder  - RHip
    (9,  10), # LHip       - RHip
    (9,  11), # LHip       - LKnee
    (11, 13), # LKnee      - LAnkle
    (13, 15), # LAnkle     - LBigToe
    (13, 17), # LAnkle     - LHeel
    (10, 12), # RHip       - RKnee
    (12, 14), # RKnee      - RAnkle
    (14, 18), # RAnkle     - RBigToe
    (14, 20), # RAnkle     - RHeel
    (27, 28), # LAcromion  - RAcromion
    (27, 29), # LAcromion  - Neck
    (28, 29), # RAcromion  - Neck
    (0,  29), # Nose       - Neck
]


# ─────────────────────────────────────────────────────────────────────────────
# OpenSim MarkerSet .osim writer
# ─────────────────────────────────────────────────────────────────────────────

def write_markerset_osim(filepath: str | Path) -> None:
    """Write an OpenSim MarkerSet .osim file for all 27 body markers.

    Uses gait2392 body segment names and the <body> tag format compatible with
    OpenSim 4.x Scale Tool and IK Tool.  Marker names match TRC column headers.
    """
    MARKER_BODIES = {
        "Nose":          "head",
        "Neck":          "torso",
        "LAcromion":     "torso",
        "RAcromion":     "torso",
        "LShoulder":     "humerus_l",
        "RShoulder":     "humerus_r",
        "LOlecranon":    "ulna_l",
        "ROlecranon":    "ulna_r",
        "LElbow":        "ulna_l",
        "RElbow":        "ulna_r",
        "LWrist":        "hand_l",
        "RWrist":        "hand_r",
        "LHip":          "pelvis",
        "RHip":          "pelvis",
        "LKnee":         "tibia_l",
        "RKnee":         "tibia_r",
        "LAnkle":        "talus_l",
        "RAnkle":        "talus_r",
        "LHeel":         "calcn_l",
        "RHeel":         "calcn_r",
        "LBigToe":       "toes_l",
        "LSmallToe":     "toes_l",
        "RBigToe":       "toes_r",
        "RSmallToe":     "toes_r",
        "PelvisCenter":  "pelvis",
        "Thorax":        "torso",
        "SpineMid":      "torso",
    }

    lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<OpenSimDocument Version="40000">',
        '\t<MarkerSet name="FastSAM3DBody_markers">',
        '\t\t<objects>',
    ]
    for name in BODY_MARKER_NAMES:
        body = MARKER_BODIES.get(name, "torso")
        lines += [
            f'\t\t\t<Marker name="{name}">',
            f'\t\t\t\t<body>{body}</body>',
            '\t\t\t\t<location>0 0 0</location>',
            '\t\t\t\t<fixed>false</fixed>',
            '\t\t\t</Marker>',
        ]
    lines += [
        '\t\t</objects>',
        '\t</MarkerSet>',
        '</OpenSimDocument>',
    ]
    Path(filepath).write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MOT writer (body joint angles from 3-D positions)
# ─────────────────────────────────────────────────────────────────────────────

def _vec_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two 3-D vectors."""
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _signed_angle_axis(v1, v2, axis):
    """Signed angle from v1 to v2 around axis (all 3-D)."""
    cross = np.cross(v1, v2)
    sin = np.dot(cross, axis / (np.linalg.norm(axis) + 1e-9))
    cos = np.dot(v1, v2)
    return float(np.degrees(np.arctan2(sin, cos)))


def compute_joint_angles(markers: np.ndarray) -> dict:
    """Compute approximate anatomical joint angles from Y-up marker positions.

    Markers order follows BODY_MARKER_NAMES (indices 0-26).
    Returns degrees (positive = anatomical flexion convention where noted).
    """
    up = np.array([0.0, 1.0, 0.0])
    fwd = np.array([0.0, 0.0, 1.0])

    # Indices match BODY_MARKER_NAMES (0=Nose,1=LEye,2=REye,3=LEar,4=REar,
    # 5=LShoulder,6=RShoulder,7=LElbow,8=RElbow,9=LHip,10=RHip,
    # 11=LKnee,12=RKnee,13=LAnkle,14=RAnkle,15=LBigToe,16=LSmallToe,
    # 17=LHeel,18=RBigToe,19=RSmallToe,20=RHeel,21=RWrist,22=LWrist,…)
    l_hip    = markers[9];  r_hip    = markers[10]
    l_knee   = markers[11]; r_knee   = markers[12]
    l_ankle  = markers[13]; r_ankle  = markers[14]
    l_heel   = markers[17]; r_heel   = markers[20]
    l_btoe   = markers[15]; r_btoe   = markers[18]
    l_shldr  = markers[5];  r_shldr  = markers[6]
    l_elbow  = markers[7];  r_elbow  = markers[8]
    l_wrist  = markers[22]; r_wrist  = markers[21]
    pelvis   = (l_hip + r_hip) / 2
    mid_shldr = (l_shldr + r_shldr) / 2

    def _seg(a, b):
        d = b - a; n = np.linalg.norm(d); return d / n if n > 1e-6 else fwd.copy()

    l_knee_flex = 180.0 - _vec_angle(_seg(l_hip, l_knee), _seg(l_knee, l_ankle))
    r_knee_flex = 180.0 - _vec_angle(_seg(r_hip, r_knee), _seg(r_knee, r_ankle))

    lateral = _seg(l_hip, r_hip)
    sagittal_normal = np.cross(up, lateral); sagittal_normal /= np.linalg.norm(sagittal_normal) + 1e-9
    l_thigh = _seg(l_hip, l_knee)
    r_thigh = _seg(r_hip, r_knee)
    l_hip_flex = _signed_angle_axis(-up, l_thigh, lateral)
    r_hip_flex = _signed_angle_axis(-up, r_thigh, -lateral)
    l_hip_abd  = _signed_angle_axis(-up, l_thigh, sagittal_normal)
    r_hip_abd  = _signed_angle_axis(-up, r_thigh, -sagittal_normal)

    l_foot  = _seg(l_heel, l_btoe); r_foot  = _seg(r_heel, r_btoe)
    l_shank = _seg(l_knee, l_ankle); r_shank = _seg(r_knee, r_ankle)
    l_ankle_df = 90.0 - _vec_angle(l_shank, l_foot)
    r_ankle_df = 90.0 - _vec_angle(r_shank, r_foot)

    l_uarm = _seg(l_shldr, l_elbow); l_farm = _seg(l_elbow, l_wrist)
    r_uarm = _seg(r_shldr, r_elbow); r_farm = _seg(r_elbow, r_wrist)
    l_elbow_flex = 180.0 - _vec_angle(l_uarm, l_farm)
    r_elbow_flex = 180.0 - _vec_angle(r_uarm, r_farm)

    trunk = _seg(pelvis, mid_shldr)
    trunk_flex = _signed_angle_axis(up, trunk, lateral)
    trunk_lat  = _signed_angle_axis(up, trunk, sagittal_normal)

    return {
        "pelvis_tx":     float(pelvis[0]),
        "pelvis_ty":     float(pelvis[1]),
        "pelvis_tz":     float(pelvis[2]),
        "l_hip_flexion": l_hip_flex,
        "l_hip_adduction": -l_hip_abd,
        "r_hip_flexion": r_hip_flex,
        "r_hip_adduction": -r_hip_abd,
        "l_knee_flexion": l_knee_flex,
        "r_knee_flexion": r_knee_flex,
        "l_ankle_dorsiflexion": l_ankle_df,
        "r_ankle_dorsiflexion": r_ankle_df,
        "l_elbow_flexion": l_elbow_flex,
        "r_elbow_flexion": r_elbow_flex,
        "trunk_flexion":  trunk_flex,
        "trunk_lateral":  trunk_lat,
    }


def write_mot(
    filepath: str | Path,
    timestamps: List[float],
    frames_markers: List[np.ndarray | None],
) -> None:
    """Write an OpenSim MOT file with approximate anatomical joint angles."""
    filepath = Path(filepath)
    N = len(timestamps)

    last_good = np.zeros((len(BODY_MARKER_NAMES), 3), dtype=np.float64)
    filled = []
    for m in frames_markers:
        if m is not None and not np.any(np.isnan(m)):
            last_good = m.astype(np.float64)
        filled.append(last_good.copy())

    angle_rows = [compute_joint_angles(m) for m in filled]
    col_names = list(angle_rows[0].keys())

    with open(filepath, "w") as f:
        f.write(f"{filepath.stem}\n")
        f.write("version=1\n")
        f.write(f"nRows={N}\n")
        f.write(f"nColumns={1 + len(col_names)}\n")
        f.write("inDegrees=yes\n")
        f.write("\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(col_names) + "\n")
        for ts, row in zip(timestamps, angle_rows):
            vals = "\t".join(f"{row[c]:.6f}" for c in col_names)
            f.write(f"{ts:.6f}\t{vals}\n")


# ─────────────────────────────────────────────────────────────────────────────
# GLB animated skeleton writer — skeletal skinning (O(N × N_joints), not O(N²))
# ─────────────────────────────────────────────────────────────────────────────

def _glb_pack(gltf_dict: dict, bin_data: bytes) -> bytes:
    """Pack a GLTF dict + binary blob into a GLB binary."""
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * pad_j
    pad_b = (4 - len(bin_data) % 4) % 4
    bin_data_padded = bin_data + b'\x00' * pad_b
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack("<II", len(bin_data_padded), 0x004E4942) + bin_data_padded
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))
    return header + json_chunk + bin_chunk


def write_skeleton_glb(
    filepath: str | Path,
    timestamps: List[float],
    frames_markers: List[np.ndarray | None],
    links: List[tuple] = SKELETON_LINKS,
) -> None:
    """Write an animated skeleton as a GLB file using GLTF2 skeletal skinning.

    Animation cost is O(N_frames × N_joints) — suitable for full-length videos.

    Parameters
    ----------
    filepath : output .glb path
    timestamps : frame timestamps in seconds
    frames_markers : per-frame [N_markers, 3] Y-up arrays (or None)
    links : list of (i, j) index pairs defining bone segments
    """
    filepath = Path(filepath)
    # Infer N_joints from actual data (not hardcoded BODY_MARKER_NAMES)
    first_valid = next((m for m in frames_markers if m is not None), None)
    N_joints = first_valid.shape[0] if first_valid is not None else len(BODY_MARKER_NAMES)

    last_good = np.zeros((N_joints, 3), dtype=np.float32)
    filled = []
    for m in frames_markers:
        if m is not None and not np.any(np.isnan(m)):
            last_good = m.astype(np.float32)
        filled.append(last_good.copy())

    N_frames = len(filled)
    bind_pos  = filled[0].copy()

    bone_indices = np.array(
        [[a, b] for (a, b) in links], dtype=np.uint16
    ).flatten()

    joints_attr = np.zeros((N_joints, 4), dtype=np.uint8)
    for i in range(N_joints):
        joints_attr[i, 0] = i
    weights_attr = np.zeros((N_joints, 4), dtype=np.float32)
    weights_attr[:, 0] = 1.0

    ibm = np.zeros((N_joints, 16), dtype=np.float32)
    for i, p in enumerate(bind_pos):
        ibm[i] = [1,0,0,0, 0,1,0,0, 0,0,1,0, -p[0],-p[1],-p[2],1]

    all_translations = np.stack(
        [(f - bind_pos) for f in filled], axis=1
    ).astype(np.float32)   # [N_joints, N_frames, 3]

    anim_times = np.array([float(t) for t in timestamps], dtype=np.float32)

    chunks = []
    byte_offset = 0

    def _add(data: bytes):
        nonlocal byte_offset
        pad = (4 - len(data) % 4) % 4
        data += b'\x00' * pad
        chunks.append(data)
        start = byte_offset
        byte_offset += len(data)
        return start, len(data) - pad

    off_pos, len_pos = _add(bind_pos.tobytes())
    off_idx, len_idx = _add(bone_indices.tobytes())
    off_jnt, len_jnt = _add(joints_attr.tobytes())
    off_wgt, len_wgt = _add(weights_attr.tobytes())
    off_ibm, len_ibm = _add(ibm.tobytes())
    off_t,   len_t   = _add(anim_times.tobytes())

    trans_offsets, trans_lens = [], []
    for i in range(N_joints):
        o, l_ = _add(all_translations[i].tobytes())
        trans_offsets.append(o)
        trans_lens.append(l_)

    bin_data = b''.join(chunks)

    def _b3(a): m, M = a.min(axis=0).tolist(), a.max(axis=0).tolist(); return m, M
    def _b1(a): return [float(a.min())], [float(a.max())]

    bufferViews = [
        {"buffer": 0, "byteOffset": off_pos, "byteLength": len_pos, "target": 34962},
        {"buffer": 0, "byteOffset": off_idx, "byteLength": len_idx, "target": 34963},
        {"buffer": 0, "byteOffset": off_jnt, "byteLength": len_jnt, "target": 34962},
        {"buffer": 0, "byteOffset": off_wgt, "byteLength": len_wgt, "target": 34962},
        {"buffer": 0, "byteOffset": off_ibm, "byteLength": len_ibm},
        {"buffer": 0, "byteOffset": off_t,   "byteLength": len_t},
    ]
    BV_POS, BV_IDX, BV_JNT, BV_WGT, BV_IBM, BV_T = range(6)

    p_min, p_max = _b3(bind_pos)
    t_min, t_max = _b1(anim_times)

    accessors = [
        {"bufferView": BV_POS, "byteOffset": 0, "componentType": 5126,
         "count": N_joints, "type": "VEC3", "min": p_min, "max": p_max},
        {"bufferView": BV_IDX, "byteOffset": 0, "componentType": 5123,
         "count": len(bone_indices), "type": "SCALAR"},
        {"bufferView": BV_JNT, "byteOffset": 0, "componentType": 5121,
         "count": N_joints, "type": "VEC4"},
        {"bufferView": BV_WGT, "byteOffset": 0, "componentType": 5126,
         "count": N_joints, "type": "VEC4"},
        {"bufferView": BV_IBM, "byteOffset": 0, "componentType": 5126,
         "count": N_joints, "type": "MAT4"},
        {"bufferView": BV_T, "byteOffset": 0, "componentType": 5126,
         "count": N_frames, "type": "SCALAR", "min": t_min, "max": t_max},
    ]
    ACC_T = 5
    trans_acc_start = len(accessors)

    for i in range(N_joints):
        bv_idx = len(bufferViews)
        bufferViews.append({"buffer": 0, "byteOffset": trans_offsets[i],
                             "byteLength": trans_lens[i]})
        tr = all_translations[i]
        tr_min, tr_max = _b3(tr)
        accessors.append({
            "bufferView": bv_idx, "byteOffset": 0, "componentType": 5126,
            "count": N_frames, "type": "VEC3",
            "min": tr_min, "max": tr_max,
        })

    MESH_NODE = N_joints
    joint_nodes = [
        {"name": BODY_MARKER_NAMES[i], "translation": bind_pos[i].tolist()}
        for i in range(N_joints)
    ]
    mesh_node = {"mesh": 0, "skin": 0}

    anim_samplers, anim_channels = [], []
    for i in range(N_joints):
        s = len(anim_samplers)
        anim_samplers.append({"input": ACC_T, "output": trans_acc_start + i,
                               "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s,
                               "target": {"node": i, "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "FastSAM3DBody OpenSim Exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(N_joints + 1))}],
        "nodes": joint_nodes + [mesh_node],
        "meshes": [{
            "name": "skeleton",
            "primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 1,
                "mode": 1,
            }],
        }],
        "skins": [{
            "name": "skeleton_skin",
            "joints": list(range(N_joints)),
            "inverseBindMatrices": 4,
        }],
        "animations": [{
            "name": "take",
            "samplers": anim_samplers,
            "channels": anim_channels,
        }],
        "accessors": accessors,
        "bufferViews": bufferViews,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    filepath.write_bytes(_glb_pack(gltf, bin_data))


# ─────────────────────────────────────────────────────────────────────────────
# Mesh GLB writer (full body mesh animation using morph targets)
# ─────────────────────────────────────────────────────────────────────────────

# ── Joint / bone overlay constants ───────────────────────────────────────────
# (keypoint_index, side)  side: 'R'=orange, 'L'=green, 'C'=blue
_MESH_JOINT_MARKERS = [
    # Body
    (0,  'C', 0.018),   # nose/head
    (5,  'L', 0.020),   # left_shoulder
    (6,  'R', 0.020),   # right_shoulder
    (7,  'L', 0.017),   # left_elbow
    (8,  'R', 0.017),   # right_elbow
    (9,  'L', 0.020),   # left_hip
    (10, 'R', 0.020),   # right_hip
    (11, 'L', 0.018),   # left_knee
    (12, 'R', 0.018),   # right_knee
    (13, 'L', 0.015),   # left_ankle
    (14, 'R', 0.015),   # right_ankle
    (17, 'L', 0.012),   # left_heel
    (20, 'R', 0.012),   # right_heel
    (15, 'L', 0.010),   # left_big_toe
    (16, 'L', 0.009),   # left_small_toe
    (18, 'R', 0.010),   # right_big_toe
    (19, 'R', 0.009),   # right_small_toe
    (69, 'C', 0.016),   # neck
    # Hands (full inference mode)
    (41, 'R', 0.013),   # right_wrist
    (62, 'L', 0.013),   # left_wrist
    (21, 'R', 0.008),   # right_thumb_tip
    (25, 'R', 0.008),   # right_index_tip
    (29, 'R', 0.008),   # right_middle_tip
    (33, 'R', 0.008),   # right_ring_tip
    (37, 'R', 0.008),   # right_pinky_tip
    (24, 'R', 0.007),   # right_thumb_knuckle
    (28, 'R', 0.007),   # right_index_knuckle
    (32, 'R', 0.007),   # right_middle_knuckle
    (36, 'R', 0.007),   # right_ring_knuckle
    (40, 'R', 0.007),   # right_pinky_knuckle
    (42, 'L', 0.008),   # left_thumb_tip
    (46, 'L', 0.008),   # left_index_tip
    (50, 'L', 0.008),   # left_middle_tip
    (54, 'L', 0.008),   # left_ring_tip
    (58, 'L', 0.008),   # left_pinky_tip
    (45, 'L', 0.007),   # left_thumb_knuckle
    (49, 'L', 0.007),   # left_index_knuckle
    (53, 'L', 0.007),   # left_middle_knuckle
    (57, 'L', 0.007),   # left_ring_knuckle
    (61, 'L', 0.007),   # left_pinky_knuckle
    # Head
    (3,  'L', 0.010),   # left ear
    (4,  'R', 0.010),   # right ear
    (1,  'L', 0.008),   # left eye
    (2,  'R', 0.008),   # right eye
    # Spine + head points (indices 70–76, appended in write_mesh_glb)
    # Real joints from pred_joint_coords; geometric fallback if unavailable
    (70, 'C', 0.014),   # pelvis center (geometric)
    (71, 'C', 0.011),   # c_spine0 / lower lumbar
    (72, 'C', 0.011),   # c_spine1 / upper lumbar
    (73, 'C', 0.011),   # c_spine2 / lower thoracic
    (74, 'C', 0.011),   # c_spine3 / upper thoracic
    (75, 'C', 0.012),   # c_neck
    (76, 'C', 0.012),   # c_head
]
_MESH_BONE_PAIRS = [
    # Arms
    (5,  7),    # L shoulder → elbow
    (7,  62),   # L elbow → wrist
    (6,  8),    # R shoulder → elbow
    (8,  41),   # R elbow → wrist
    # Legs
    (9,  11),   # L hip → knee
    (11, 13),   # L knee → ankle
    (10, 12),   # R hip → knee
    (12, 14),   # R knee → ankle
    # Feet
    (13, 15),   # L ankle → big_toe
    (13, 16),   # L ankle → small_toe
    (13, 17),   # L ankle → heel
    (14, 18),   # R ankle → big_toe
    (14, 19),   # R ankle → small_toe
    (14, 20),   # R ankle → heel
    # Torso / spine  (70=PelvisCenter, 71=c_spine0, 72=c_spine1, 73=c_spine2, 74=c_spine3)
    (5,  6),    # shoulders across
    (9,  10),   # hips across
    (9,  70),   # L hip → pelvis center
    (10, 70),   # R hip → pelvis center
    (70, 71),   # pelvis center → c_spine0
    (71, 72),   # c_spine0 → c_spine1
    (72, 73),   # c_spine1 → c_spine2
    (73, 74),   # c_spine2 → c_spine3
    (74, 5),    # c_spine3 → L shoulder
    (74, 6),    # c_spine3 → R shoulder
    # Neck  (75=c_neck)
    (75, 74),   # c_neck → c_spine3
    (69, 75),   # surface neck → c_neck
    # Head  (76=c_head)
    (76, 75),   # c_head → c_neck
    (76, 1),    # c_head → L eye
    (76, 2),    # c_head → R eye
    # Right hand fingers
    (41, 24),   # wrist → thumb_knuckle
    (24, 23), (23, 22), (22, 21),        # thumb
    (41, 28),   # wrist → index_knuckle
    (28, 27), (27, 26), (26, 25),        # index
    (41, 32),   # wrist → middle_knuckle
    (32, 31), (31, 30), (30, 29),        # middle
    (41, 36),   # wrist → ring_knuckle
    (36, 35), (35, 34), (34, 33),        # ring
    (41, 40),   # wrist → pinky_knuckle
    (40, 39), (39, 38), (38, 37),        # pinky
    # Left hand fingers
    (62, 45),   # wrist → thumb_knuckle
    (45, 44), (44, 43), (43, 42),        # thumb
    (62, 49),   # wrist → index_knuckle
    (49, 48), (48, 47), (47, 46),        # index
    (62, 53),   # wrist → middle_knuckle
    (53, 52), (52, 51), (51, 50),        # middle
    (62, 57),   # wrist → ring_knuckle
    (57, 56), (56, 55), (55, 54),        # ring
    (62, 61),   # wrist → pinky_knuckle
    (61, 60), (60, 59), (59, 58),        # pinky
]
_SIDE_COLORS = {
    'R': [1.00, 0.45, 0.08, 1.0],
    'L': [0.08, 0.85, 0.25, 1.0],
    'C': [0.20, 0.50, 1.00, 1.0],
}


def _sphere_verts_faces(radius: float, segments: int = 8, rings: int = 6):
    """Return (verts float32 [N,3], faces uint32 [M*3])."""
    import math
    verts = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments):
            theta = 2 * math.pi * s / segments
            verts.append([
                radius * math.sin(phi) * math.cos(theta),
                radius * math.cos(phi),
                radius * math.sin(phi) * math.sin(theta),
            ])
    faces = []
    for r in range(rings):
        for s in range(segments):
            ns = (s + 1) % segments
            a = r * segments + s
            b = r * segments + ns
            c = (r + 1) * segments + s
            d = (r + 1) * segments + ns
            faces.extend([a, c, b, b, c, d])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)


def _cylinder_verts_faces(radius: float = 0.010, segments: int = 6):
    """Unit cylinder (length=1) along Y from -0.5 to +0.5."""
    import math
    verts = []
    for cap in range(2):
        y = -0.5 + cap
        for s in range(segments):
            a = 2 * math.pi * s / segments
            verts.append([radius * math.cos(a), y, radius * math.sin(a)])
    verts.append([0.0, -0.5, 0.0])   # bottom cap centre
    verts.append([0.0,  0.5, 0.0])   # top cap centre
    faces = []
    for s in range(segments):
        ns = (s + 1) % segments
        a, b, c, d = s, ns, segments + s, segments + ns
        faces.extend([a, c, b, b, c, d])
    bot_c = 2 * segments
    top_c = 2 * segments + 1
    for s in range(segments):
        ns = (s + 1) % segments
        faces.extend([bot_c, ns, s])
        faces.extend([top_c, segments + s, segments + ns])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)


def _quat_y_to_dir(d: np.ndarray) -> np.ndarray:
    """glTF quaternion [x,y,z,w] rotating local Y to direction d."""
    import math
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    d = d / n
    dot = float(d[1])          # dot(Y, d)
    cross = np.array([d[2], 0.0, -d[0]], dtype=np.float64)    # Y × d = (dz, 0, -dx)
    cn = np.linalg.norm(cross)
    if cn < 1e-6:
        if dot > 0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cos_h = math.sqrt(max(0.0, (1.0 + dot) / 2.0))
    sin_h = math.sqrt(max(0.0, (1.0 - dot) / 2.0))
    axis = cross / cn
    return np.array([axis[0] * sin_h, axis[1] * sin_h, axis[2] * sin_h, cos_h],
                    dtype=np.float32)


_HAND_KPT_INDICES = set(range(21, 63))   # finger joints + wrists (21-40 R, 41 R-wrist, 42-61 L, 62 L-wrist)

def write_mesh_glb(
    filepath: str | Path,
    timestamps: List[float],
    frames_verts: List[np.ndarray | None],
    faces: np.ndarray,
    frames_kpts: List[np.ndarray | None] | None = None,
    frames_cam_t: List[np.ndarray | None] | None = None,
    frames_joint_coords: List[np.ndarray | None] | None = None,
    body_only: bool = False,
) -> None:
    """Write animated full body mesh as GLB using morph targets.

    Parameters
    ----------
    filepath : output .glb
    timestamps : frame timestamps in seconds
    frames_verts : per-frame [18439, 3] vertex arrays in camera space (or None)
    faces : [N_faces, 3] int triangle indices (from estimator.faces)
    frames_kpts : per-frame [70, 3] camera-space keypoints (no cam_t), or None
    frames_cam_t : per-frame [3] camera translation, or None
    frames_joint_coords : per-frame [127, 3] camera-space MHR joint coords (no cam_t), or None
    body_only : if True, skip all hand/wrist joint markers and bone sticks
    """
    filepath = Path(filepath)

    # ── Fill mesh frames (forward-fill missing) ───────────────────────────────
    last_good = None
    filled = []
    for v in frames_verts:
        if v is not None and not np.any(np.isnan(v)):
            verts_yup = v.copy().astype(np.float32)
            verts_yup[:, 1] = -verts_yup[:, 1]   # Y-up
            verts_yup[:, 0] = -verts_yup[:, 0]   # fix mirror (camera X = subject's left)
            last_good = verts_yup
        filled.append(last_good.copy() if last_good is not None else None)

    if not any(f is not None for f in filled):
        return

    N_verts = faces.max() + 1
    zero_v = np.zeros((N_verts, 3), dtype=np.float32)
    filled = [f if f is not None else zero_v for f in filled]

    N_frames = len(filled)
    N_morphs = N_frames - 1
    faces_np = np.asarray(faces, dtype=np.uint32).flatten()

    # ── Build per-frame world-space keypoints (aligned with mesh) ─────────────
    has_kpts = (frames_kpts is not None and frames_cam_t is not None)
    kpts_world: List[np.ndarray | None] = [None] * N_frames
    if has_kpts:
        last_kpts: np.ndarray | None = None
        for i, (k, ct) in enumerate(zip(frames_kpts, frames_cam_t)):
            if k is not None and ct is not None and not np.any(np.isnan(k)):
                w = (k + ct[None, :]).astype(np.float32)
                w[:, 1] = -w[:, 1]   # Y-up
                w[:, 0] = -w[:, 0]   # fix mirror
                last_kpts = w
            kpts_world[i] = last_kpts.copy() if last_kpts is not None else None

    # ── Build per-frame world-space joint coords (127-joint MHR armature) ─────
    has_jcoords = (frames_joint_coords is not None and frames_cam_t is not None)
    jcoords_world: List[np.ndarray | None] = [None] * N_frames
    if has_jcoords:
        last_jc: np.ndarray | None = None
        for i, (jc, ct) in enumerate(zip(frames_joint_coords, frames_cam_t)):
            if jc is not None and ct is not None and not np.any(np.isnan(jc)):
                w = (jc + ct[None, :]).astype(np.float32)
                w[:, 1] = -w[:, 1]   # Y-up
                w[:, 0] = -w[:, 0]   # fix mirror
                last_jc = w
            jcoords_world[i] = last_jc.copy() if last_jc is not None else None

    # ── Center at pelvis each frame (remove global translation) ───────────────
    if has_kpts:
        last_pelvis = np.zeros(3, dtype=np.float32)
        for i in range(N_frames):
            kw = kpts_world[i]
            if kw is not None:
                last_pelvis = ((kw[9] + kw[10]) / 2.0).astype(np.float32)
            filled[i] = filled[i] - last_pelvis[None, :]
            if kpts_world[i] is not None:
                kpts_world[i] = kpts_world[i] - last_pelvis[None, :]
            if has_jcoords and jcoords_world[i] is not None:
                jcoords_world[i] = jcoords_world[i] - last_pelvis[None, :]

    # ── Smooth mesh vertices + keypoints (Butterworth 6 Hz, same as PostProcessor) ──
    _fps_est = float(N_frames - 1) / max(float(timestamps[-1] - timestamps[0]), 1e-3)
    _nyq = _fps_est / 2.0
    _cut = 6.0
    if _cut < _nyq and N_frames >= 13:
        from scipy.signal import butter, filtfilt
        _b, _a = butter(4, _cut / _nyq, btype='low')
        _flat = np.stack(filled, axis=0).reshape(N_frames, -1)
        filled = [r.reshape(-1, 3) for r in filtfilt(_b, _a, _flat, axis=0).astype(np.float32)]
        if has_kpts:
            _kw_ref = next(kw for kw in kpts_world if kw is not None)
            _kw_fill = [kw if kw is not None else _kw_ref for kw in kpts_world]
            _kw_arr = np.stack(_kw_fill, axis=0).reshape(N_frames, -1)
            _kw_smooth = filtfilt(_b, _a, _kw_arr, axis=0).astype(np.float32)
            kpts_world = [
                _kw_smooth[i].reshape(-1, 3) if kpts_world[i] is not None else None
                for i in range(N_frames)
            ]
        if has_jcoords:
            _jc_non_none = [jc for jc in jcoords_world if jc is not None]
            if _jc_non_none:
                _jc_ref = _jc_non_none[0]
                _jc_fill = [jc if jc is not None else _jc_ref for jc in jcoords_world]
                _jc_arr = np.stack(_jc_fill, axis=0).reshape(N_frames, -1)
                _jc_smooth = filtfilt(_b, _a, _jc_arr, axis=0).astype(np.float32)
                jcoords_world = [
                    _jc_smooth[i].reshape(-1, 3) if jcoords_world[i] is not None else None
                    for i in range(N_frames)
                ]

    # ── Append spine/head points at indices 70–76 ─────────────────────────────
    # 70=PelvisCenter(geometric), 71=c_spine0, 72=c_spine1, 73=c_spine2,
    # 74=c_spine3, 75=c_neck, 76=c_head
    # Uses real pred_joint_coords joints; falls back to geometric interpolation.
    if has_kpts:
        for i in range(N_frames):
            kw = kpts_world[i]
            if kw is not None:
                pelvis_c = ((kw[9] + kw[10]) / 2.0).astype(np.float32)  # idx 70
                if has_jcoords and jcoords_world[i] is not None:
                    jc = jcoords_world[i]
                    spine0 = jc[34]    # c_spine0 (lower lumbar)   idx 71
                    spine1 = jc[35]    # c_spine1 (upper lumbar)   idx 72
                    spine2 = jc[36]    # c_spine2 (lower thoracic) idx 73
                    spine3 = jc[37]    # c_spine3 (upper thoracic) idx 74
                    neck   = jc[110]   # c_neck                    idx 75
                    head   = jc[113]   # c_head                    idx 76
                else:
                    # Geometric fallback (no joint_coords available)
                    thorax = ((kw[5] + kw[6]) / 2.0).astype(np.float32)
                    spine0 = (pelvis_c * 0.75 + thorax * 0.25).astype(np.float32)
                    spine1 = (pelvis_c * 0.50 + thorax * 0.50).astype(np.float32)
                    spine2 = (pelvis_c * 0.25 + thorax * 0.75).astype(np.float32)
                    spine3 = thorax
                    neck   = kw[69].astype(np.float32)   # surface neck kpt
                    head   = ((kw[3] + kw[4]) / 2.0).astype(np.float32)  # ear midpoint
                kpts_world[i] = np.vstack([kw, pelvis_c, spine0, spine1, spine2, spine3, neck, head])

    base_pos = filled[0]

    # ── Binary buffer helpers ─────────────────────────────────────────────────
    def _pack_f32(a): return np.asarray(a, dtype=np.float32).tobytes()
    def _pack_f16(a): return np.asarray(a, dtype=np.float16).tobytes()
    def _pack_u32(a): return np.asarray(a, dtype=np.uint32).tobytes()
    def _pack_i16_delta(delta):
        """Quantize (N,3) float32 delta to INT16 normalized, return (bytes, min, max)."""
        dmin = delta.min(axis=0); dmax = delta.max(axis=0)
        scale = np.maximum(np.maximum(np.abs(dmin), np.abs(dmax)), 1e-9)
        raw = np.clip(np.round(delta / scale * 32767), -32767, 32767).astype(np.int16)
        return raw.tobytes(), dmin.tolist(), dmax.tolist()

    byte_offset = 0
    chunks: list[bytes] = []

    def _add(data: bytes):
        nonlocal byte_offset
        pad = (4 - len(data) % 4) % 4
        data += b'\x00' * pad
        chunks.append(data)
        start = byte_offset
        byte_offset += len(data)
        return start, len(data) - pad

    def _bounds(arr):
        return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()

    # ── Body mesh geometry ────────────────────────────────────────────────────
    off0, len0 = _add(_pack_f32(base_pos))
    off_i, len_i = _add(_pack_u32(faces_np))

    morph_deltas, morph_offsets, morph_lens = [], [], []
    for f in filled[1:]:
        delta = (f - base_pos).astype(np.float32)
        packed, dmin, dmax = _pack_i16_delta(delta)
        o, l_ = _add(packed)
        morph_offsets.append(o); morph_lens.append(l_); morph_deltas.append((dmin, dmax))

    # ── Shared timestamp accessor ─────────────────────────────────────────────
    anim_times = np.array([float(t) for t in timestamps], dtype=np.float32)
    off_t, len_t = _add(_pack_f32(anim_times))

    # ── Morph weights ─────────────────────────────────────────────────────────
    weights_np = np.zeros((N_frames, max(N_morphs, 1)), dtype=np.float32)
    for f in range(1, N_frames):
        weights_np[f, f - 1] = 1.0
    off_w, len_w = _add(_pack_f32(weights_np))

    # ── Overlay geometry: sphere + cylinder ───────────────────────────────────
    sph_v, sph_f = _sphere_verts_faces(radius=1.0, segments=8, rings=5)
    cyl_v, cyl_f = _cylinder_verts_faces(radius=1.0, segments=6)
    off_sv, len_sv = _add(_pack_f32(sph_v))
    off_sf, len_sf = _add(_pack_u32(sph_f))
    off_cv, len_cv = _add(_pack_f32(cyl_v))
    off_cf, len_cf = _add(_pack_u32(cyl_f))

    # ── Per-joint translation data ────────────────────────────────────────────
    # Body decoder predicts all hand/wrist keypoints even in body-only mode — show them all
    joint_markers = list(_MESH_JOINT_MARKERS)
    joint_trans_data: list[tuple[int, np.ndarray]] = []   # (kpt_idx, [N,3])
    if has_kpts:
        for kpt_idx, side, radius in joint_markers:
            trans = np.zeros((N_frames, 3), dtype=np.float32)
            for fi, kw in enumerate(kpts_world):
                if kw is not None and kpt_idx < kw.shape[0]:
                    trans[fi] = kw[kpt_idx]
                elif fi > 0:
                    trans[fi] = trans[fi - 1]
            joint_trans_data.append((kpt_idx, trans))

    # ── Per-bone translation / rotation / scale data ──────────────────────────
    _FINGER_INDICES = set(range(21, 62))
    bone_pairs = list(_MESH_BONE_PAIRS)
    bone_trs_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if has_kpts:
        for a_idx, b_idx in bone_pairs:
            stick_r = 0.004 if (a_idx in _FINGER_INDICES or b_idx in _FINGER_INDICES) else 0.007
            tr = np.zeros((N_frames, 3), dtype=np.float32)
            ro = np.zeros((N_frames, 4), dtype=np.float32)
            sc = np.ones((N_frames, 3), dtype=np.float32)
            ro[:, 3] = 1.0   # identity quaternion w=1
            for fi, kw in enumerate(kpts_world):
                if kw is not None and max(a_idx, b_idx) < kw.shape[0]:
                    pa = kw[a_idx]; pb = kw[b_idx]
                    tr[fi] = (pa + pb) * 0.5
                    d = pb - pa
                    length = float(np.linalg.norm(d))
                    ro[fi] = _quat_y_to_dir(d)
                    sc[fi] = [stick_r, length, stick_r]
                elif fi > 0:
                    tr[fi] = tr[fi - 1]
                    ro[fi] = ro[fi - 1]
                    sc[fi] = sc[fi - 1]
            bone_trs_data.append((tr, ro, sc))

    # ── Pack overlay animation data ───────────────────────────────────────────
    joint_trans_accs: list[int] = []
    for _, trans in joint_trans_data:
        o, l_ = _add(_pack_f32(trans))
        joint_trans_accs.append((o, l_, len(joint_trans_data[0][1])))

    bone_trans_accs: list[tuple] = []
    bone_rot_accs:   list[tuple] = []
    bone_scale_accs: list[tuple] = []
    for tr, ro, sc in bone_trs_data:
        ot, lt = _add(_pack_f32(tr)); bone_trans_accs.append((ot, lt))
        or_, lr = _add(_pack_f32(ro)); bone_rot_accs.append((or_, lr))
        os, ls = _add(_pack_f32(sc)); bone_scale_accs.append((os, ls))

    bin_data = b''.join(chunks)

    # ── Build glTF JSON ───────────────────────────────────────────────────────
    bufferViews = [
        {"buffer": 0, "byteOffset": off0,  "byteLength": len0,  "target": 34962},
        {"buffer": 0, "byteOffset": off_i,  "byteLength": len_i,  "target": 34963},
    ]
    accessors = [
        {
            "bufferView": 0, "byteOffset": 0, "componentType": 5126,
            "count": len(base_pos), "type": "VEC3",
            **dict(zip(["min", "max"], _bounds(base_pos))),
        },
        {
            "bufferView": 1, "byteOffset": 0, "componentType": 5125,
            "count": len(faces_np), "type": "SCALAR",
        },
    ]

    morph_targets = []
    for (dmin, dmax), o, l_ in zip(morph_deltas, morph_offsets, morph_lens):
        bv_idx = len(bufferViews)
        bufferViews.append({"buffer": 0, "byteOffset": o, "byteLength": l_, "target": 34962})
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": bv_idx, "byteOffset": 0, "componentType": 5122,
            "normalized": True, "count": len(base_pos), "type": "VEC3",
            "min": dmin, "max": dmax,
        })
        morph_targets.append({"POSITION": acc_idx})

    # Timestamp accessor (shared by all channels)
    bv_t = len(bufferViews)
    bufferViews.append({"buffer": 0, "byteOffset": off_t, "byteLength": len_t})
    acc_t = len(accessors)
    accessors.append({
        "bufferView": bv_t, "byteOffset": 0, "componentType": 5126,
        "count": N_frames, "type": "SCALAR",
        "min": [float(anim_times.min())], "max": [float(anim_times.max())],
    })

    # Morph weights accessor
    bv_w = len(bufferViews)
    bufferViews.append({"buffer": 0, "byteOffset": off_w, "byteLength": len_w})
    acc_w = len(accessors)
    accessors.append({
        "bufferView": bv_w, "byteOffset": 0, "componentType": 5126,
        "count": N_frames * max(N_morphs, 1), "type": "SCALAR",
    })

    # Sphere geometry accessors
    bv_sv = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": off_sv, "byteLength": len_sv, "target": 34962})
    acc_sv = len(accessors);  accessors.append({"bufferView": bv_sv, "byteOffset": 0, "componentType": 5126, "count": len(sph_v), "type": "VEC3", **dict(zip(["min","max"], _bounds(sph_v)))})
    bv_sf = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": off_sf, "byteLength": len_sf, "target": 34963})
    acc_sf = len(accessors);  accessors.append({"bufferView": bv_sf, "byteOffset": 0, "componentType": 5125, "count": len(sph_f), "type": "SCALAR"})

    bv_cv = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": off_cv, "byteLength": len_cv, "target": 34962})
    acc_cv = len(accessors);  accessors.append({"bufferView": bv_cv, "byteOffset": 0, "componentType": 5126, "count": len(cyl_v), "type": "VEC3", **dict(zip(["min","max"], _bounds(cyl_v)))})
    bv_cf = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": off_cf, "byteLength": len_cf, "target": 34963})
    acc_cf = len(accessors);  accessors.append({"bufferView": bv_cf, "byteOffset": 0, "componentType": 5125, "count": len(cyl_f), "type": "SCALAR"})

    # Per-joint translation accessors
    joint_acc_t: list[int] = []
    for o, l_, cnt in joint_trans_accs:
        bv = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": o, "byteLength": l_})
        ac = len(accessors);   accessors.append({"bufferView": bv, "byteOffset": 0, "componentType": 5126, "count": N_frames, "type": "VEC3"})
        joint_acc_t.append(ac)

    # Per-bone translation / rotation / scale accessors
    bone_acc_t: list[int] = []
    bone_acc_r: list[int] = []
    bone_acc_s: list[int] = []
    for ot, lt in bone_trans_accs:
        bv = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": ot, "byteLength": lt})
        ac = len(accessors);   accessors.append({"bufferView": bv, "byteOffset": 0, "componentType": 5126, "count": N_frames, "type": "VEC3"})
        bone_acc_t.append(ac)
    for or_, lr in bone_rot_accs:
        bv = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": or_, "byteLength": lr})
        ac = len(accessors);   accessors.append({"bufferView": bv, "byteOffset": 0, "componentType": 5126, "count": N_frames, "type": "VEC4"})
        bone_acc_r.append(ac)
    for os, ls in bone_scale_accs:
        bv = len(bufferViews); bufferViews.append({"buffer": 0, "byteOffset": os, "byteLength": ls})
        ac = len(accessors);   accessors.append({"bufferView": bv, "byteOffset": 0, "componentType": 5126, "count": N_frames, "type": "VEC3"})
        bone_acc_s.append(ac)

    # ── Materials ─────────────────────────────────────────────────────────────
    materials = [
        {   # 0: body skin — translucent blue
            "name": "body_skin",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.45, 0.62, 0.82, 0.38],
                "roughnessFactor": 0.70, "metallicFactor": 0.0,
            },
            "alphaMode": "BLEND", "doubleSided": True,
        },
        {   # 1: right joint — orange
            "name": "joint_right",
            "pbrMetallicRoughness": {"baseColorFactor": [1.00, 0.45, 0.08, 1.0], "roughnessFactor": 0.22, "metallicFactor": 0.15},
        },
        {   # 2: left joint — green
            "name": "joint_left",
            "pbrMetallicRoughness": {"baseColorFactor": [0.08, 0.85, 0.25, 1.0], "roughnessFactor": 0.22, "metallicFactor": 0.15},
        },
        {   # 3: centre joint — blue
            "name": "joint_center",
            "pbrMetallicRoughness": {"baseColorFactor": [0.20, 0.50, 1.00, 1.0], "roughnessFactor": 0.22, "metallicFactor": 0.15},
        },
        {   # 4: bone stick — ivory
            "name": "bone_ivory",
            "pbrMetallicRoughness": {"baseColorFactor": [0.90, 0.86, 0.74, 1.0], "roughnessFactor": 0.55, "metallicFactor": 0.05},
        },
    ]
    _side_mat = {'R': 1, 'L': 2, 'C': 3}

    # ── Sphere meshes (one per joint, different scale encodes radius) ──────────
    # We use scale [r,r,r] per node to set per-joint radius at runtime.
    # One shared sphere mesh (material varies by side — create 3 sphere meshes).
    sph_meshes: dict[str, int] = {}   # side -> mesh index
    meshes_list = [
        {
            "name": "body",
            "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4,
                            "material": 0, "targets": morph_targets}],
            "weights": [0.0] * max(N_morphs, 1),
        }
    ]
    for side, mat_idx in _side_mat.items():
        mesh_idx = len(meshes_list)
        sph_meshes[side] = mesh_idx
        meshes_list.append({
            "name": f"sphere_{side}",
            "primitives": [{"attributes": {"POSITION": acc_sv}, "indices": acc_sf, "mode": 4, "material": mat_idx}],
        })
    cyl_mesh_idx = len(meshes_list)
    meshes_list.append({
        "name": "bone_cyl",
        "primitives": [{"attributes": {"POSITION": acc_cv}, "indices": acc_cf, "mode": 4, "material": 4}],
    })

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # node 0: body mesh
    nodes_list = [{"mesh": 0}]
    scene_nodes = [0]

    joint_node_indices: list[int] = []
    for i, (kpt_idx, side, radius) in enumerate(joint_markers):
        if not has_kpts:
            break
        node_idx = len(nodes_list)
        nodes_list.append({
            "mesh": sph_meshes[side],
            "scale": [radius, radius, radius],
        })
        scene_nodes.append(node_idx)
        joint_node_indices.append(node_idx)

    bone_node_indices: list[int] = []
    for i in range(len(bone_pairs)):
        if not has_kpts:
            break
        node_idx = len(nodes_list)
        nodes_list.append({"mesh": cyl_mesh_idx})
        scene_nodes.append(node_idx)
        bone_node_indices.append(node_idx)

    # ── Animation samplers + channels ─────────────────────────────────────────
    anim_samplers = [
        {"input": acc_t, "output": acc_w, "interpolation": "STEP"},          # morph weights
    ]
    anim_channels = [
        {"sampler": 0, "target": {"node": 0, "path": "weights"}},
    ]

    for i, (node_idx, acc_tr) in enumerate(zip(joint_node_indices, joint_acc_t)):
        s_idx = len(anim_samplers)
        anim_samplers.append({"input": acc_t, "output": acc_tr, "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s_idx, "target": {"node": node_idx, "path": "translation"}})

    for i, node_idx in enumerate(bone_node_indices):
        s_t = len(anim_samplers)
        anim_samplers.append({"input": acc_t, "output": bone_acc_t[i], "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s_t, "target": {"node": node_idx, "path": "translation"}})
        s_r = len(anim_samplers)
        anim_samplers.append({"input": acc_t, "output": bone_acc_r[i], "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s_r, "target": {"node": node_idx, "path": "rotation"}})
        s_s = len(anim_samplers)
        anim_samplers.append({"input": acc_t, "output": bone_acc_s[i], "interpolation": "LINEAR"})
        anim_channels.append({"sampler": s_s, "target": {"node": node_idx, "path": "scale"}})

    # ── Assemble glTF ─────────────────────────────────────────────────────────
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "FastSAM3DBody OpenSim Exporter"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes_list,
        "meshes": meshes_list,
        "materials": materials,
        "animations": [{"name": "take", "samplers": anim_samplers, "channels": anim_channels}],
        "accessors": accessors,
        "bufferViews": bufferViews,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    # ── Write GLB ─────────────────────────────────────────────────────────────
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * pad_j
    pad_b = (4 - len(bin_data) % 4) % 4
    bin_data += b'\x00' * pad_b

    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack("<II", len(bin_data),   0x004E4942) + bin_data
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))

    filepath.write_bytes(header + json_chunk + bin_chunk)
    n_joints = len(joint_node_indices)
    n_bones  = len(bone_node_indices)
    print(f"  Mesh GLB: {n_joints} joint spheres, {n_bones} bone sticks, translucent skin")
