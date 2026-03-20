# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
OpenSim export utilities: TRC, MOT, and GLB from MHR inference output.

Coordinate system
-----------------
MHR outputs are in camera space:
  X: rightward in image
  Y: downward in image
  Z: into screen (depth)

World-space position = pred_keypoints_3d + pred_cam_t[None, :]

OpenSim / TRC convention (Y-up):
  X_os = cam_X   (lateral, right)
  Y_os = -cam_Y  (vertical, up)
  Z_os = cam_Z   (depth, forward into scene)

Units: metres (the MHR model predicts in metres).

TRC marker subset
-----------------
22 anatomical landmarks from the MHR-70 set (body only, no fingers):
  nose(0), l_shoulder(5), r_shoulder(6), l_elbow(7), r_elbow(8),
  l_hip(9), r_hip(10), l_knee(11), r_knee(12), l_ankle(13), r_ankle(14),
  l_big_toe(15), l_small_toe(16), l_heel(17), r_big_toe(18),
  r_small_toe(19), r_heel(20), r_wrist(41), l_wrist(62),
  l_olecranon(63), r_olecranon(64), l_acromion(67), r_acromion(68), neck(69)
"""

from __future__ import annotations

import struct
import json
import base64
from pathlib import Path
from typing import List

import numpy as np

# ── Body marker subset (index into the 70-joint MHR keypoint array) ──────────
BODY_MARKER_IDX = [
    0,   # nose
    5,   # left_shoulder
    6,   # right_shoulder
    7,   # left_elbow
    8,   # right_elbow
    9,   # left_hip
    10,  # right_hip
    11,  # left_knee
    12,  # right_knee
    13,  # left_ankle
    14,  # right_ankle
    15,  # left_big_toe
    16,  # left_small_toe
    17,  # left_heel
    18,  # right_big_toe
    19,  # right_small_toe
    20,  # right_heel
    41,  # right_wrist
    62,  # left_wrist
    63,  # left_olecranon
    64,  # right_olecranon
    67,  # left_acromion
    68,  # right_acromion
    69,  # neck
]

BODY_MARKER_NAMES = [
    "nose",
    "l_shoulder", "r_shoulder",
    "l_elbow",    "r_elbow",
    "l_hip",      "r_hip",
    "l_knee",     "r_knee",
    "l_ankle",    "r_ankle",
    "l_big_toe",  "l_small_toe", "l_heel",
    "r_big_toe",  "r_small_toe", "r_heel",
    "r_wrist",    "l_wrist",
    "l_olecranon", "r_olecranon",
    "l_acromion",  "r_acromion",
    "neck",
]

assert len(BODY_MARKER_IDX) == len(BODY_MARKER_NAMES)

# Skeleton connectivity for GLB line visualization (marker index pairs)
SKELETON_LINKS = [
    (1, 2),   # l_shoulder - r_shoulder
    (1, 3),   # l_shoulder - l_elbow
    (3, 18),  # l_elbow    - l_wrist
    (2, 4),   # r_shoulder - r_elbow
    (4, 17),  # r_elbow    - r_wrist
    (1, 5),   # l_shoulder - l_hip
    (2, 6),   # r_shoulder - r_hip
    (5, 6),   # l_hip      - r_hip
    (5, 7),   # l_hip      - l_knee
    (7, 9),   # l_knee     - l_ankle
    (9, 11),  # l_ankle    - l_big_toe
    (9, 13),  # l_ankle    - l_heel
    (6, 8),   # r_hip      - r_knee
    (8, 10),  # r_knee     - r_ankle
    (10, 14), # r_ankle    - r_big_toe
    (10, 16), # r_ankle    - r_heel
    (21, 22), # l_acromion - r_acromion
    (21, 23), # l_acromion - neck
    (22, 23), # r_acromion - neck
    (0, 23),  # nose       - neck
]


def cam_to_opensim(pts: np.ndarray) -> np.ndarray:
    """Convert camera-space points [N, 3] to OpenSim Y-up [N, 3].

    camera: X-right, Y-down, Z-forward
    opensim: X-right, Y-up, Z-forward
    """
    out = pts.copy()
    out[:, 1] = -pts[:, 1]
    return out


def extract_body_markers(person: dict) -> np.ndarray | None:
    """Return [N_markers, 3] world-space Y-up marker positions or None."""
    kpts = person.get("pred_keypoints_3d")   # [70, 3] body-local
    cam_t = person.get("pred_cam_t")          # [3]
    if kpts is None or cam_t is None:
        return None
    if np.any(np.isnan(kpts)) or np.any(np.isnan(cam_t)):
        return None
    world = kpts + cam_t[None, :]              # camera-space
    markers = world[BODY_MARKER_IDX]           # [N_markers, 3]
    return cam_to_opensim(markers)             # Y-up


# ─────────────────────────────────────────────────────────────────────────────
# TRC writer
# ─────────────────────────────────────────────────────────────────────────────

def write_trc(
    filepath: str | Path,
    timestamps: List[float],
    frames_markers: List[np.ndarray | None],
    marker_names: List[str] = BODY_MARKER_NAMES,
    units: str = "m",
) -> None:
    """Write an OpenSim TRC file.

    Parameters
    ----------
    filepath : path for the .trc file
    timestamps : list of frame timestamps in seconds (length N_frames)
    frames_markers : list of [N_markers, 3] arrays or None for missing frames
    marker_names : marker label strings
    units : 'm' for metres (OpenSim default)
    """
    filepath = Path(filepath)
    N = len(timestamps)
    M = len(marker_names)
    data_rate = 1.0 / (timestamps[1] - timestamps[0]) if N > 1 else 30.0

    # Fill missing frames by last known pose (forward fill)
    last_good = np.zeros((M, 3), dtype=np.float64)
    filled = []
    for m in frames_markers:
        if m is not None and not np.any(np.isnan(m)):
            last_good = m.astype(np.float64)
        filled.append(last_good.copy())

    with open(filepath, "w") as f:
        # Header rows
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath.name}\n")
        f.write(
            f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            f"OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )
        f.write(
            f"{data_rate:.4f}\t{data_rate:.4f}\t{N}\t{M}\t{units}\t"
            f"{data_rate:.4f}\t1\t{N}\n"
        )
        # Marker names row: Frame# | Time | Name1 | | | Name2 | | | ...
        names_row = "Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\t\t"
        f.write(names_row + "\n")
        # Coordinate labels row
        coord_labels = "\t\t" + "\t".join(
            [f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(M)]
        )
        f.write(coord_labels + "\n")
        f.write("\n")  # blank line (some TRC readers expect this)

        # Data rows
        for fi, (ts, m) in enumerate(zip(timestamps, filled)):
            row = f"{fi+1}\t{ts:.6f}"
            for j in range(M):
                row += f"\t{m[j,0]:.6f}\t{m[j,1]:.6f}\t{m[j,2]:.6f}"
            f.write(row + "\n")


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

    Markers order follows BODY_MARKER_NAMES:
      0:nose, 1:l_shoulder, 2:r_shoulder, 3:l_elbow, 4:r_elbow,
      5:l_hip, 6:r_hip, 7:l_knee, 8:r_knee, 9:l_ankle, 10:r_ankle,
      11:l_big_toe, 12:l_small_toe, 13:l_heel, 14:r_big_toe,
      15:r_small_toe, 16:r_heel, 17:r_wrist, 18:l_wrist,
      19:l_olecranon, 20:r_olecranon, 21:l_acromion, 22:r_acromion, 23:neck

    Returns degrees (positive = anatomical flexion convention where noted).
    """
    up = np.array([0.0, 1.0, 0.0])
    fwd = np.array([0.0, 0.0, 1.0])  # Z is depth in our convention

    # Anatomical directions
    l_hip    = markers[5];  r_hip    = markers[6]
    l_knee   = markers[7];  r_knee   = markers[8]
    l_ankle  = markers[9];  r_ankle  = markers[10]
    l_heel   = markers[13]; r_heel   = markers[16]
    l_btoe   = markers[11]; r_btoe   = markers[14]
    l_shldr  = markers[1];  r_shldr  = markers[2]
    l_elbow  = markers[3];  r_elbow  = markers[4]
    l_wrist  = markers[18]; r_wrist  = markers[17]
    pelvis   = (l_hip + r_hip) / 2
    mid_shldr = (l_shldr + r_shldr) / 2

    def _seg(a, b):
        d = b - a; n = np.linalg.norm(d); return d / n if n > 1e-6 else fwd.copy()

    # Knee flexion: angle between thigh and shank (0 = straight, 90 = 90° flexed)
    l_knee_flex = 180.0 - _vec_angle(_seg(l_hip, l_knee), _seg(l_knee, l_ankle))
    r_knee_flex = 180.0 - _vec_angle(_seg(r_hip, r_knee), _seg(r_knee, r_ankle))

    # Hip flexion: angle of thigh from vertical (in sagittal plane)
    lateral = _seg(l_hip, r_hip)
    sagittal_normal = np.cross(up, lateral); sagittal_normal /= np.linalg.norm(sagittal_normal) + 1e-9
    l_thigh = _seg(l_hip, l_knee)
    r_thigh = _seg(r_hip, r_knee)
    l_hip_flex = _signed_angle_axis(-up, l_thigh, lateral)
    r_hip_flex = _signed_angle_axis(-up, r_thigh, -lateral)

    # Hip abduction: lateral deviation of thigh from vertical in frontal plane
    l_hip_abd = _signed_angle_axis(-up, l_thigh, sagittal_normal)
    r_hip_abd = _signed_angle_axis(-up, r_thigh, -sagittal_normal)

    # Ankle dorsiflexion: angle between shank and foot (90° = neutral)
    l_foot = _seg(l_heel, l_btoe)
    r_foot = _seg(r_heel, r_btoe)
    l_shank = _seg(l_knee, l_ankle)
    r_shank = _seg(r_knee, r_ankle)
    l_ankle_df = 90.0 - _vec_angle(l_shank, l_foot)
    r_ankle_df = 90.0 - _vec_angle(r_shank, r_foot)

    # Elbow flexion
    l_uarm = _seg(l_shldr, l_elbow); l_farm = _seg(l_elbow, l_wrist)
    r_uarm = _seg(r_shldr, r_elbow); r_farm = _seg(r_elbow, r_wrist)
    l_elbow_flex = 180.0 - _vec_angle(l_uarm, l_farm)
    r_elbow_flex = 180.0 - _vec_angle(r_uarm, r_farm)

    # Trunk lean (forward/backward): angle of trunk from vertical
    trunk = _seg(pelvis, mid_shldr)
    trunk_flex = _signed_angle_axis(up, trunk, lateral)
    trunk_lat  = _signed_angle_axis(up, trunk, sagittal_normal)

    # Pelvis tx/ty/tz: position of midpoint of hips (translation)
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

    # Forward-fill missing frames
    last_good = np.zeros((len(BODY_MARKER_NAMES), 3), dtype=np.float64)
    filled = []
    for m in frames_markers:
        if m is not None and not np.any(np.isnan(m)):
            last_good = m.astype(np.float64)
        filled.append(last_good.copy())

    # Compute angles for each frame
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
    Each landmark is a skinning "bone" whose translation is animated; the line-
    segment mesh rigidly follows its corresponding bone.

    Parameters
    ----------
    filepath : output .glb path
    timestamps : frame timestamps in seconds
    frames_markers : per-frame [N_markers, 3] Y-up arrays (or None)
    links : list of (i, j) index pairs defining bone segments
    """
    filepath = Path(filepath)
    N_joints = len(BODY_MARKER_NAMES)

    # ── Forward-fill missing frames ───────────────────────────────────────────
    last_good = np.zeros((N_joints, 3), dtype=np.float32)
    filled = []
    for m in frames_markers:
        if m is not None and not np.any(np.isnan(m)):
            last_good = m.astype(np.float32)
        filled.append(last_good.copy())

    N_frames = len(filled)
    bind_pos  = filled[0].copy()   # [N_joints, 3] bind-pose world positions

    # ── Skeleton line-mesh ────────────────────────────────────────────────────
    bone_indices = np.array(
        [[a, b] for (a, b) in links], dtype=np.uint16
    ).flatten()

    # Each vertex i is weighted 100 % to joint i
    joints_attr = np.zeros((N_joints, 4), dtype=np.uint8)
    for i in range(N_joints):
        joints_attr[i, 0] = i
    weights_attr = np.zeros((N_joints, 4), dtype=np.float32)
    weights_attr[:, 0] = 1.0

    # Inverse bind matrices: 24 × 4×4 column-major float32
    # Vertex i starts at bind_pos[i]; IBM = translation(-bind_pos[i])
    ibm = np.zeros((N_joints, 16), dtype=np.float32)
    for i, p in enumerate(bind_pos):
        ibm[i] = [1,0,0,0, 0,1,0,0, 0,0,1,0, -p[0],-p[1],-p[2],1]

    # Per-joint translation animation: delta from bind pose so that
    #   skinned_pos = joint_current - bind_pos + bind_pos = joint_current
    all_translations = np.stack(
        [(f - bind_pos) for f in filled], axis=1
    ).astype(np.float32)   # [N_joints, N_frames, 3]

    anim_times = np.array([float(t) for t in timestamps], dtype=np.float32)

    # ── Build binary buffer ───────────────────────────────────────────────────
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

    # ── GLTF JSON ─────────────────────────────────────────────────────────────
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
        # 0: vertex positions (bind)
        {"bufferView": BV_POS, "byteOffset": 0, "componentType": 5126,
         "count": N_joints, "type": "VEC3", "min": p_min, "max": p_max},
        # 1: line indices
        {"bufferView": BV_IDX, "byteOffset": 0, "componentType": 5123,
         "count": len(bone_indices), "type": "SCALAR"},
        # 2: JOINTS_0
        {"bufferView": BV_JNT, "byteOffset": 0, "componentType": 5121,
         "count": N_joints, "type": "VEC4"},
        # 3: WEIGHTS_0
        {"bufferView": BV_WGT, "byteOffset": 0, "componentType": 5126,
         "count": N_joints, "type": "VEC4"},
        # 4: inverse bind matrices
        {"bufferView": BV_IBM, "byteOffset": 0, "componentType": 5126,
         "count": N_joints, "type": "MAT4"},
        # 5: animation timestamps
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
                "mode": 1,  # LINES
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
# Mesh GLB writer (full body mesh animation using trimesh + pygltflib)
# ─────────────────────────────────────────────────────────────────────────────

def write_mesh_glb(
    filepath: str | Path,
    timestamps: List[float],
    frames_verts: List[np.ndarray | None],
    faces: np.ndarray,
) -> None:
    """Write animated full body mesh as GLB using morph targets.

    For long sequences, set --max_frames or use --skip_mesh to keep file sizes
    manageable (18 439 verts × 4 bytes × 3 × N_frames).

    Parameters
    ----------
    filepath : output .glb
    timestamps : frame timestamps in seconds
    frames_verts : per-frame [18439, 3] vertex arrays in Y-up metres (or None)
    faces : [N_faces, 3] int triangle indices (from estimator.faces)
    """
    filepath = Path(filepath)

    # Forward-fill
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

    # Replace None with zeros
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
