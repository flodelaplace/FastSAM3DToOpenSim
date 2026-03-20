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

def write_mesh_glb(
    filepath: str | Path,
    timestamps: List[float],
    frames_verts: List[np.ndarray | None],
    faces: np.ndarray,
) -> None:
    """Write animated full body mesh as GLB using morph targets.

    Parameters
    ----------
    filepath : output .glb
    timestamps : frame timestamps in seconds
    frames_verts : per-frame [18439, 3] vertex arrays in camera space (or None)
    faces : [N_faces, 3] int triangle indices (from estimator.faces)
    """
    filepath = Path(filepath)

    last_good = None
    filled = []
    for v in frames_verts:
        if v is not None and not np.any(np.isnan(v)):
            verts_yup = v.copy().astype(np.float32)
            verts_yup[:, 1] = -verts_yup[:, 1]   # Y-up
            last_good = verts_yup
        if last_good is None:
            filled.append(None)
        else:
            filled.append(last_good.copy())

    if not any(f is not None for f in filled):
        return

    N_verts = faces.max() + 1
    zero_v = np.zeros((N_verts, 3), dtype=np.float32)
    filled = [f if f is not None else zero_v for f in filled]

    N_frames = len(filled)
    N_morphs = N_frames - 1
    base_pos = filled[0]
    faces_np = np.asarray(faces, dtype=np.uint32).flatten()

    def _pack_f32(a): return a.astype(np.float32).tobytes()
    def _pack_u32(a): return a.astype(np.uint32).tobytes()

    byte_offset = 0
    chunks = []

    def _add(data: bytes):
        nonlocal byte_offset
        pad = (4 - len(data) % 4) % 4
        data += b'\x00' * pad
        chunks.append(data)
        start = byte_offset
        byte_offset += len(data)
        return start, len(data) - pad

    off0, len0 = _add(_pack_f32(base_pos))
    off_i, len_i = _add(_pack_u32(faces_np))

    morph_offsets, morph_lens, morph_deltas = [], [], []
    for f in filled[1:]:
        delta = f - base_pos
        o, l_ = _add(_pack_f32(delta))
        morph_offsets.append(o); morph_lens.append(l_); morph_deltas.append(delta)

    anim_times = np.array([float(t) for t in timestamps], dtype=np.float32)
    off_t, len_t = _add(_pack_f32(anim_times))

    weights_np = np.zeros((N_frames, max(N_morphs, 1)), dtype=np.float32)
    for f in range(1, N_frames):
        weights_np[f, f - 1] = 1.0
    off_w, len_w = _add(_pack_f32(weights_np))

    bin_data = b''.join(chunks)

    def _bounds(arr):
        return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()

    bufferViews = [
        {"buffer": 0, "byteOffset": off0, "byteLength": len0, "target": 34962},
        {"buffer": 0, "byteOffset": off_i, "byteLength": len_i, "target": 34963},
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
    for delta, o, l_ in zip(morph_deltas, morph_offsets, morph_lens):
        bv_idx = len(bufferViews)
        bufferViews.append({"buffer": 0, "byteOffset": o, "byteLength": l_, "target": 34962})
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": bv_idx, "byteOffset": 0, "componentType": 5126,
            "count": len(base_pos), "type": "VEC3",
            **dict(zip(["min", "max"], _bounds(delta))),
        })
        morph_targets.append({"POSITION": acc_idx})

    bv_t = len(bufferViews)
    bufferViews.append({"buffer": 0, "byteOffset": off_t, "byteLength": len_t})
    acc_t = len(accessors)
    accessors.append({
        "bufferView": bv_t, "byteOffset": 0, "componentType": 5126,
        "count": N_frames, "type": "SCALAR",
        "min": [float(anim_times.min())], "max": [float(anim_times.max())],
    })

    bv_w = len(bufferViews)
    bufferViews.append({"buffer": 0, "byteOffset": off_w, "byteLength": len_w})
    acc_w = len(accessors)
    accessors.append({
        "bufferView": bv_w, "byteOffset": 0, "componentType": 5126,
        "count": N_frames * max(N_morphs, 1), "type": "SCALAR",
    })

    gltf = {
        "asset": {"version": "2.0", "generator": "FastSAM3DBody OpenSim Exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "name": "body",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                        "targets": morph_targets,
                    }
                ],
                "weights": [0.0] * max(N_morphs, 1),
            }
        ],
        "animations": [
            {
                "name": "take",
                "samplers": [{"input": acc_t, "output": acc_w, "interpolation": "STEP"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
            }
        ],
        "accessors": accessors,
        "bufferViews": bufferViews,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * pad_j
    pad_b = (4 - len(bin_data) % 4) % 4
    bin_data += b'\x00' * pad_b

    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack("<II", len(bin_data), 0x004E4942) + bin_data
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))

    filepath.write_bytes(header + json_chunk + bin_chunk)
