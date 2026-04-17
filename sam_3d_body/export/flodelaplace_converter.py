"""
MHR → Rajagopal mocap markerset converter.

Produces 64 markers matching assets/rajagopal2015_mocap.osim:
  - Direct 70-kpt indices        : head, body joint centers, acromions, feet,
                                    wrists, hand (metacarpals + tips)
  - Direct 127-jcoord indices    : spine (c_spine0..3, c_neck, c_head),
                                    RCLAV/LCLAV, HTOP
  - Cached mesh vertex indices   : ASIS/PSIS, femoral condyles (LFC/MFC),
                                    malleoli (LMAL/MMAL), humeral epicondyles
                                    (LEL/MEL), wrist styloids (FAradius/FAulna),
                                    C7 (loaded from
                                    assets/flodelaplace_anatomical_vertex_idx.json)

Vertex indices were chosen once on a reference frame (Squat.MP4) using
tools/viz_rajagopal_markers.py. MHR mesh topology is deterministic so they
transfer across subjects and frames.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

_ASSETS = Path(__file__).resolve().parents[2] / "assets"
_VERTEX_IDX_JSON = _ASSETS / "flodelaplace_anatomical_vertex_idx.json"

# ── Direct MHR 70-kpt mappings ────────────────────────────────────────────────
_KPT70: dict[str, int] = {
    # Head
    "Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
    # Body joint centers — LSJC/RSJC dropped: MHR kpt 5/6 labeled "shoulder"
    # do not reliably land on the glenohumeral joint center (acromion is used
    # instead for the shoulder segment in IK/scaling).
    "LEJC": 7,  "REJC": 8,
    "LHJC": 9,  "RHJC": 10,
    "LKJC": 11, "RKJC": 12,
    "LAJC": 13, "RAJC": 14,
    # Feet
    "LTOE": 15, "LMT5": 16, "LCAL": 17,
    "RTOE": 18, "RMT5": 19, "RCAL": 20,
    # Hand metacarpals (MCP, match the Coco133 positions set in the .osim).
    # MHR labels kpt-X as the tip and kpt-X+3 as the most proximal joint
    # (MCP). Verified empirically on Squat.MP4:
    #   RThumb:  kpt 21 = tip 12.7cm from wrist, kpt 24 = MCP 4.1cm
    #   RIndex:  kpt 25 = tip 16.1cm,            kpt 28 = MCP 8.5cm
    #   RPinky:  kpt 37 = tip 13.6cm,            kpt 40 = MCP 7.5cm
    "RThumb": 24, "RIndex": 28, "RPinky": 40,
    "LThumb": 45, "LIndex": 49, "LPinky": 61,
    # Hand fingertips (pose2sim-style distal positions).
    "RIndexTip": 25, "RPinkyTip": 37,
    "LIndexTip": 46, "LPinkyTip": 58,
    # Wrists & acromions
    "RWrist_hand": 41, "LWrist_hand": 62,
    "LACR": 67, "RACR": 68,
}

# ── Direct MHR 127-joint armature mappings ────────────────────────────────────
_JCOORD127: dict[str, int] = {
    "c_spine0": 34,   # lower lumbar (L5-S1)
    "c_spine1": 35,   # upper lumbar
    "c_spine2": 36,   # lower thoracic
    "c_spine3": 37,   # upper thoracic
    "c_neck":   110,  # cervical
    "c_head":   113,  # head joint
    "RCLAV":    74,   # right clavicle (validated visually (Florian Delaplace))
    "LCLAV":    38,   # left clavicle
    "HTOP":     126,  # top of head (vertex of skull)
}


def load_anatomical_vertex_indices() -> dict[str, int]:
    with open(_VERTEX_IDX_JSON) as f:
        return json.load(f)["vertex_indices"]


class FlodelaplaceConverter:
    """MHR inference outputs → Rajagopal mocap marker array."""

    def __init__(self):
        self.vertex_indices = load_anatomical_vertex_indices()
        # Preserve grouping order in the output TRC: direct kpts first,
        # then jcoord-direct, then vertex picks. Within each group the
        # declaration order is kept.
        self.marker_names: List[str] = (
            list(_KPT70.keys())
            + list(_JCOORD127.keys())
            + list(self.vertex_indices.keys())
        )

    def extract_anatomical(self, vertices_3d: np.ndarray) -> np.ndarray:
        """Index the pre-picked anatomical vertices out of the full mesh.

        Args:
            vertices_3d: (N, 18439, 3) or (18439, 3)

        Returns:
            (N, N_anat, 3) or (N_anat, 3) where N_anat = len(vertex_indices).
            Row order matches iteration order of self.vertex_indices (same as
            the tail of self.marker_names).
        """
        single = vertices_3d.ndim == 2
        if single:
            vertices_3d = vertices_3d[np.newaxis]
        idxs = np.asarray(list(self.vertex_indices.values()), dtype=np.int64)
        out = vertices_3d[:, idxs, :]
        return out[0] if single else out

    def convert(
        self,
        keypoints_3d:   np.ndarray,    # (N, 70, 3)  or (70, 3)
        jcoords_3d:     np.ndarray,    # (N, 127, 3) or (127, 3)
        anat_verts_3d:  np.ndarray,    # (N, N_anat, 3) — pre-extracted,
                                       # already in the same world frame as kpts.
    ) -> Tuple[np.ndarray, List[str]]:
        """Return (markers, marker_names).

        markers shape: (N, M, 3) or (M, 3) matching the input dim, where
        M = len(self.marker_names).  All three inputs must share the same
        world frame.  The expected caller pattern is:

            conv = FlodelaplaceConverter()
            anat_raw   = conv.extract_anatomical(verts_world_cam)    # (N, 21, 3)
            # concat anat_raw into jcoords_stack, run transformer.transform(),
            # then split out the transformed anat_verts on return.
            markers, names = conv.convert(kpts_opensim, jc_opensim, anat_opensim)
        """
        single = keypoints_3d.ndim == 2
        if single:
            keypoints_3d   = keypoints_3d[np.newaxis]
            jcoords_3d     = jcoords_3d[np.newaxis]
            anat_verts_3d  = anat_verts_3d[np.newaxis]

        marker_list = []
        for idx in _KPT70.values():
            marker_list.append(keypoints_3d[:, idx, :])
        for idx in _JCOORD127.values():
            marker_list.append(jcoords_3d[:, idx, :])
        # anat_verts_3d already ordered to match self.vertex_indices.
        for i in range(anat_verts_3d.shape[1]):
            marker_list.append(anat_verts_3d[:, i, :])

        markers = np.stack(marker_list, axis=1)  # (N, M, 3)
        if single:
            markers = markers[0]
        return markers, self.marker_names

    def get_marker_names(self) -> List[str]:
        return list(self.marker_names)
