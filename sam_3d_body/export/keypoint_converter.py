"""
MHR70 to OpenSim marker conversion.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


# MHR70 hand keypoint indices (21-40 right hand, 42-61 left hand).
# Body-only mode includes 2 markers per hand: Index tip + Pinky tip.
# These span the full width of the palm and are sufficient to constrain the
# 2 wrist DOFs (wrist_flex_r/l, wrist_dev_r/l). The Wholebody model has no
# finger joint DOFs — additional finger markers would add noise without benefit.
# Tips:  R(25=Index, 37=Pinky) + L mirrors (46=LIndex, 58=LPinky)
_HAND_TIP_INDICES = {
    25, 37,   # right: Index tip, Pinky tip
    46, 58,   # left:  Index tip, Pinky tip
}
_HAND_INDICES = (set(range(21, 41)) | set(range(42, 62))) - _HAND_TIP_INDICES

# MHR 127-joint armature indices for spine/neck/head
_SPINE_JCOORDS = {
    "c_spine0":  34,   # lower lumbar
    "c_spine1":  35,   # upper lumbar
    "c_spine2":  36,   # lower thoracic
    "c_spine3":  37,   # upper thoracic
    "c_neck":   110,   # cervical / neck joint
    "c_head":   113,   # head joint (crown)
}


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
       21: "RThumb",     22: "RThumb1",  23: "RThumb2",  24: "RThumb3",
       25: "RIndex",     26: "RIndex1",  27: "RIndex2",  28: "RIndex3",
       29: "RMiddleTip", 30: "RMiddle1", 31: "RMiddle2", 32: "RMiddle3",
       33: "RRingTip",   34: "RRing1",   35: "RRing2",   36: "RRing3",
       37: "RPinky",     38: "RPinky1",  39: "RPinky2",  40: "RPinky3",
       42: "LThumb",     43: "LThumb1",  44: "LThumb2",  45: "LThumb3",
       46: "LIndex",     47: "LIndex1",  48: "LIndex2",  49: "LIndex3",
       50: "LMiddleTip", 51: "LMiddle1", 52: "LMiddle2", 53: "LMiddle3",
       54: "LRingTip",   55: "LRing1",   56: "LRing2",   57: "LRing3",
       58: "LPinky",     59: "LPinky1",  60: "LPinky2",  61: "LPinky3",
    }

    DEFAULT_DERIVED = {
        "PelvisCenter": {"type": "midpoint", "points": [9, 10]},
        "Thorax":       {"type": "midpoint", "points": [67, 68]},
    }

    def __init__(self):
        self.opensim_names = self.DEFAULT_OPENSIM_NAMES.copy()
        self.derived_markers = self.DEFAULT_DERIVED.copy()

    def convert(
        self,
        keypoints_3d: np.ndarray,
        jcoords_3d: Optional[np.ndarray] = None,
        include_derived: bool = True,
        body_only: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert MHR70 keypoints to OpenSim markers.

        Args:
            keypoints_3d : (N, 70, 3) or (70, 3)
            jcoords_3d   : (N, 127, 3) or (127, 3) MHR armature joints (optional).
                           When provided, spine/neck/head markers are appended.
            include_derived: Add PelvisCenter and Thorax midpoint markers.
            body_only: Exclude hand markers (use when --inference_type body).

        Returns:
            (markers array, marker_names list)
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis]
            if jcoords_3d is not None:
                jcoords_3d = jcoords_3d[np.newaxis]

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

        if jcoords_3d is not None:
            for name, jidx in _SPINE_JCOORDS.items():
                marker_list.append(jcoords_3d[:, jidx, :])
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
        raise ValueError(f"Unknown derived marker type: {t}")

    def get_marker_names(self, include_derived=True, body_only=True, include_spine=True):
        names = [
            self.opensim_names[idx]
            for idx in sorted(self.opensim_names.keys())
            if not (body_only and idx in _HAND_INDICES)
        ]
        if include_derived:
            names.extend(self.derived_markers.keys())
        if include_spine:
            names.extend(_SPINE_JCOORDS.keys())
        return names
