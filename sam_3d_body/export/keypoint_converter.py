"""
MHR70 to OpenSim marker conversion.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


# MHR70 hand keypoint indices (21-40 right hand, 42-61 left hand)
_HAND_INDICES = set(range(21, 41)) | set(range(42, 62))


class KeypointConverter:
    DEFAULT_OPENSIM_NAMES = {
        0: "Nose",
        1: "LEye",      2: "REye",
        3: "LEar",      4: "REar",
        5: "LShoulder", 6: "RShoulder",
        7: "LElbow",    8: "RElbow",
        9: "LHip",     10: "RHip",
       11: "LKnee",    12: "RKnee",
       13: "LAnkle",   14: "RAnkle",
       15: "LBigToe",  16: "LSmallToe", 17: "LHeel",
       18: "RBigToe",  19: "RSmallToe", 20: "RHeel",
       41: "RWrist",   62: "LWrist",
       63: "LOlecranon", 64: "ROlecranon",
       65: "LCubitalFossa", 66: "RCubitalFossa",
       67: "LAcromion",  68: "RAcromion",
       69: "Neck",
        # Hand markers (only used in full/hand inference mode)
       21: "RThumbTip",  22: "RThumb1",  23: "RThumb2",  24: "RThumb3",
       25: "RIndexTip",  26: "RIndex1",  27: "RIndex2",  28: "RIndex3",
       29: "RMiddleTip", 30: "RMiddle1", 31: "RMiddle2", 32: "RMiddle3",
       33: "RRingTip",   34: "RRing1",   35: "RRing2",   36: "RRing3",
       37: "RPinkyTip",  38: "RPinky1",  39: "RPinky2",  40: "RPinky3",
       42: "LThumbTip",  43: "LThumb1",  44: "LThumb2",  45: "LThumb3",
       46: "LIndexTip",  47: "LIndex1",  48: "LIndex2",  49: "LIndex3",
       50: "LMiddleTip", 51: "LMiddle1", 52: "LMiddle2", 53: "LMiddle3",
       54: "LRingTip",   55: "LRing1",   56: "LRing2",   57: "LRing3",
       58: "LPinkyTip",  59: "LPinky1",  60: "LPinky2",  61: "LPinky3",
    }

    DEFAULT_DERIVED = {
        "PelvisCenter": {"type": "midpoint", "points": [9, 10]},
        "Thorax":       {"type": "midpoint", "points": [67, 68]},
        "SpineMid":     {"type": "interpolate", "points": [9, 10, 67, 68], "ratio": 0.5},
    }

    def __init__(self):
        self.opensim_names = self.DEFAULT_OPENSIM_NAMES.copy()
        self.derived_markers = self.DEFAULT_DERIVED.copy()

    def convert(
        self,
        keypoints_3d: np.ndarray,
        include_derived: bool = True,
        body_only: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert MHR70 keypoints to OpenSim markers.

        Args:
            keypoints_3d: (N, 70, 3) or (70, 3) array
            include_derived: Add PelvisCenter, Thorax, SpineMid
            body_only: Exclude hand markers (use when --inference_type body)

        Returns:
            (markers array, marker_names list)
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        marker_list = []
        marker_names = []

        for mhr_idx in sorted(self.opensim_names.keys()):
            if body_only and mhr_idx in _HAND_INDICES:
                continue
            marker_list.append(keypoints_3d[:, mhr_idx, :])
            marker_names.append(self.opensim_names[mhr_idx])

        if include_derived:
            for name, definition in self.derived_markers.items():
                derived = self._compute_derived(keypoints_3d, definition)
                marker_list.append(derived)
                marker_names.append(name)

        markers = np.stack(marker_list, axis=1)

        if single_frame:
            markers = markers[0]

        return markers, marker_names

    def _compute_derived(self, keypoints_3d, definition):
        t = definition["type"]
        pts = definition["points"]
        if t == "midpoint":
            return (keypoints_3d[:, pts[0]] + keypoints_3d[:, pts[1]]) / 2
        elif t == "interpolate":
            ratio = definition.get("ratio", 0.5)
            p1 = (keypoints_3d[:, pts[0]] + keypoints_3d[:, pts[1]]) / 2
            p2 = (keypoints_3d[:, pts[2]] + keypoints_3d[:, pts[3]]) / 2
            return p1 * (1 - ratio) + p2 * ratio
        raise ValueError(f"Unknown derived marker type: {t}")

    def get_marker_names(self, include_derived=True, body_only=True):
        names = [
            self.opensim_names[idx]
            for idx in sorted(self.opensim_names.keys())
            if not (body_only and idx in _HAND_INDICES)
        ]
        if include_derived:
            names.extend(self.derived_markers.keys())
        return names
