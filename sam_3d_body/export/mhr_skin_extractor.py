"""Extract glTF-skin-ready data from the MHR JIT model file (mhr_model.pt).

Provides:
  - bind pose (per-joint local translation + quaternion in parent frame)
  - joint parent indices
  - inverse bind matrices (4x4 per joint, world-frame inverse)
  - per-vertex skinning (4 joint indices + 4 weights, normalized)
  - rest mesh (vertices + faces)

All in numpy. Loaded once via load_mhr_skin_data().
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MHRSkinData:
    n_joints: int
    n_verts: int
    # Bind pose (per joint, in parent frame)
    bind_translation: np.ndarray   # (J, 3)
    bind_rotation_quat: np.ndarray # (J, 4) — xyzw
    parents: np.ndarray            # (J,) int — -1 for root
    # Inverse bind matrices (world → joint local in bind pose)
    inverse_bind_matrices: np.ndarray  # (J, 4, 4)
    # Mesh
    rest_vertices: np.ndarray      # (V, 3)
    faces: np.ndarray              # (F, 3) uint32
    # Per-vertex skinning (max 4 joints/weights per vertex)
    skin_joint_indices: np.ndarray # (V, 4) uint16
    skin_weights: np.ndarray       # (V, 4) float32 — sum to 1


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (J, 4) xyzw quaternions to (J, 3, 3) rotation matrices."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = q[..., 0]**2 + q[..., 1]**2 + q[..., 2]**2 + q[..., 3]**2
    s = 2.0 / np.maximum(n, 1e-12)
    M = np.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype)
    M[..., 0, 0] = 1 - s * (y * y + z * z)
    M[..., 0, 1] = s * (x * y - z * w)
    M[..., 0, 2] = s * (x * z + y * w)
    M[..., 1, 0] = s * (x * y + z * w)
    M[..., 1, 1] = 1 - s * (x * x + z * z)
    M[..., 1, 2] = s * (y * z - x * w)
    M[..., 2, 0] = s * (x * z - y * w)
    M[..., 2, 1] = s * (y * z + x * w)
    M[..., 2, 2] = 1 - s * (x * x + y * y)
    return M


def _flat_skin_to_per_vertex(
    vert_idx: np.ndarray,
    joint_idx: np.ndarray,
    weights: np.ndarray,
    n_verts: int,
    max_joints: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flattened (vert_idx, joint_idx, weight) triplets into
    per-vertex (joints, weights) arrays of shape (V, max_joints).

    For vertices with more than max_joints influences, keep only the top
    max_joints and re-normalize. Vertices with fewer get zero-padded.
    """
    out_j = np.zeros((n_verts, max_joints), dtype=np.uint16)
    out_w = np.zeros((n_verts, max_joints), dtype=np.float32)

    # Group by vertex via sort
    order = np.argsort(vert_idx, kind="stable")
    v_sorted = vert_idx[order]
    j_sorted = joint_idx[order]
    w_sorted = weights[order]

    # Find boundaries of each vertex group
    starts = np.searchsorted(v_sorted, np.arange(n_verts), side="left")
    ends = np.searchsorted(v_sorted, np.arange(n_verts), side="right")

    for v in range(n_verts):
        s, e = int(starts[v]), int(ends[v])
        if e <= s:
            continue
        ws = w_sorted[s:e]
        js = j_sorted[s:e]
        if (e - s) > max_joints:
            top = np.argsort(-ws)[:max_joints]
            ws = ws[top]
            js = js[top]
        out_j[v, :len(js)] = js.astype(np.uint16)
        out_w[v, :len(ws)] = ws

    # Normalize so each vertex's weights sum to 1
    sums = out_w.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    out_w = (out_w / sums).astype(np.float32)
    return out_j, out_w


@lru_cache(maxsize=2)
def load_mhr_skin_data(mhr_pt_path: str) -> Optional[MHRSkinData]:
    """Load skinning + skeleton data from an MHR JIT (.pt) file.

    Returns None if the file is not loadable / doesn't have the expected buffers.
    """
    p = Path(mhr_pt_path)
    if not p.is_file():
        return None
    try:
        import torch
        mhr = torch.jit.load(str(p), map_location="cpu")
    except Exception as exc:
        print(f"[mhr_skin_extractor] failed to load {p}: {exc}")
        return None

    buf = {n: b for n, b in mhr.named_buffers()}
    required = [
        "character_torch.skeleton.joint_translation_offsets",
        "character_torch.skeleton.joint_prerotations",
        "character_torch.skeleton.joint_parents",
        "character_torch.linear_blend_skinning.inverse_bind_pose",
        "character_torch.linear_blend_skinning.skin_indices_flattened",
        "character_torch.linear_blend_skinning.skin_weights_flattened",
        "character_torch.linear_blend_skinning.vert_indices_flattened",
        "character_torch.mesh.rest_vertices",
        "character_torch.mesh.faces",
    ]
    missing = [k for k in required if k not in buf]
    if missing:
        print(f"[mhr_skin_extractor] missing buffers: {missing[:3]}...")
        return None

    bt = buf["character_torch.skeleton.joint_translation_offsets"].cpu().numpy().astype(np.float32)
    bq = buf["character_torch.skeleton.joint_prerotations"].cpu().numpy().astype(np.float32)
    parents = buf["character_torch.skeleton.joint_parents"].cpu().numpy().astype(np.int32)
    inv_bind_8 = buf["character_torch.linear_blend_skinning.inverse_bind_pose"].cpu().numpy().astype(np.float32)
    skin_j = buf["character_torch.linear_blend_skinning.skin_indices_flattened"].cpu().numpy().astype(np.int64)
    skin_w = buf["character_torch.linear_blend_skinning.skin_weights_flattened"].cpu().numpy().astype(np.float32)
    skin_v = buf["character_torch.linear_blend_skinning.vert_indices_flattened"].cpu().numpy().astype(np.int64)
    rest_v = buf["character_torch.mesh.rest_vertices"].cpu().numpy().astype(np.float32)
    faces = buf["character_torch.mesh.faces"].cpu().numpy().astype(np.uint32)

    n_joints = bt.shape[0]
    n_verts = rest_v.shape[0]

    # MHR stores everything in centimetres internally. Pred outputs are scaled
    # to metres in mhr_head.py (curr_joint_coords *= 0.01). Convert all length
    # data to metres so the GLB is in standard glTF units.
    cm_to_m = 0.01
    bt = bt * cm_to_m
    rest_v = rest_v * cm_to_m

    # Build inverse bind matrices from (translation, quaternion, scale) layout.
    # We assume xyzw quaternion convention (roma/glTF native). Translation in cm
    # → convert to m. Scale is unitless and stays as is.
    inv_t = inv_bind_8[:, 0:3] * cm_to_m
    inv_q = inv_bind_8[:, 3:7]
    inv_s = inv_bind_8[:, 7:8]
    inv_R = _quat_xyzw_to_matrix(inv_q)            # (J, 3, 3)
    inv_R_scaled = inv_R * inv_s[:, None, :]
    inv_bind_matrices = np.zeros((n_joints, 4, 4), dtype=np.float32)
    inv_bind_matrices[:, :3, :3] = inv_R_scaled
    inv_bind_matrices[:, :3, 3] = inv_t
    inv_bind_matrices[:, 3, 3] = 1.0

    # Per-vertex skinning
    skin_joint_indices, skin_weights = _flat_skin_to_per_vertex(
        skin_v, skin_j, skin_w, n_verts=n_verts, max_joints=4
    )

    return MHRSkinData(
        n_joints=n_joints,
        n_verts=n_verts,
        bind_translation=bt,
        bind_rotation_quat=bq,
        parents=parents,
        inverse_bind_matrices=inv_bind_matrices,
        rest_vertices=rest_v,
        faces=faces,
        skin_joint_indices=skin_joint_indices,
        skin_weights=skin_weights,
    )
