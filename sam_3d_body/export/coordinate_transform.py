"""
Coordinate system transformation from camera to OpenSim.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import uniform_filter1d

# jcoords index for the head joint (c_head) in the MHR 127-joint armature
_JCOORDS_HEAD_IDX = 113

# MHR70 foot indices used for ground alignment and height measurement
_FOOT_INDICES = [15, 17, 18, 20]   # LBigToe, LHeel, RBigToe, RHeel


class CoordinateTransformer:
    """
    Transforms coordinates from SAM3D Body camera space to OpenSim world space.

    Camera:  X=right, Y=down,    Z=forward(depth)
    OpenSim: X=forward(anterior), Y=up, Z=right(lateral)

    Height scaling uses the user-provided subject_height directly:
      - If jcoords are supplied, c_head (jcoords[113]) is the top reference.
      - Otherwise falls back to Nose (keypoint index 0) with a fixed ratio.
    In both cases the foot ground (min of heel/toe Y) is the bottom reference,
    giving a single exact scale with no magic correction factors.
    """

    CAMERA_TO_OPENSIM = np.array(
        [
            [0,  0, 1],   # X_opensim = Z_camera
            [0, -1, 0],   # Y_opensim = -Y_camera
            [1,  0, 0],   # Z_opensim = X_camera
        ],
        dtype=np.float64,
    )

    # Nose is at ~93.5 % of standing height; used only when c_head unavailable
    _NOSE_HEIGHT_FRACTION = 0.935

    def __init__(self, subject_height: float = 1.75, units: str = "m"):
        self.subject_height = subject_height
        self.units = units
        self.scale_factor = 1000.0 if units == "mm" else 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        keypoints_3d: np.ndarray,
        jcoords_3d: Optional[np.ndarray] = None,
        camera_translation: Optional[np.ndarray] = None,
        center_pelvis: bool = True,
        align_to_ground: bool = True,
        apply_global_translation: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform keypoints (and optionally jcoords) to OpenSim world space.

        Args:
            keypoints_3d : (N, 70, 3)  MHR70 keypoints in camera space
            jcoords_3d   : (N, 127, 3) MHR armature joints in camera space (optional)

        Returns:
            keypoints_opensim              if jcoords_3d is None
            (keypoints_opensim, jcoords_opensim)  otherwise
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis]
            if jcoords_3d is not None:
                jcoords_3d = jcoords_3d[np.newaxis]

        kpts = keypoints_3d.copy().astype(np.float64)
        jc   = jcoords_3d.copy().astype(np.float64) if jcoords_3d is not None else None

        # 1. Rotate axes: camera → OpenSim
        kpts = kpts @ self.CAMERA_TO_OPENSIM.T
        if jc is not None:
            jc = jc @ self.CAMERA_TO_OPENSIM.T

        # 2. Uniform height scale using ground-truth subject_height
        scale = self._compute_height_scale(kpts, jc)
        kpts = kpts * scale
        if jc is not None:
            jc = jc * scale

        # 3. Global translation / pelvis centering
        if apply_global_translation and camera_translation is not None:
            kpts, xz_deltas = self._apply_global_translation(kpts, camera_translation, scale)
            if jc is not None:
                jc[:, :, 0] += xz_deltas[:, 0:1]
                jc[:, :, 2] += xz_deltas[:, 2:3]
        elif center_pelvis:
            shift = self._pelvis_shifts(kpts)      # (N, 3) with Y=0
            kpts = kpts - shift[:, None, :]
            if jc is not None:
                jc = jc - shift[:, None, :]
            self._last_pelvis_shifts = shift * self.scale_factor  # saved in output units (mm)

        # 4. Align feet to Y=0 each frame
        if align_to_ground:
            kpts, ground_offsets = self._align_to_ground(kpts, return_offsets=True)
            if jc is not None:
                jc[:, :, 1] -= ground_offsets[:, None]

        # 5. Unit conversion (m → mm if requested)
        kpts = kpts * self.scale_factor
        if jc is not None:
            jc = jc * self.scale_factor

        if single_frame:
            kpts = kpts[0]
            if jc is not None:
                jc = jc[0]

        return (kpts, jc) if jcoords_3d is not None else kpts

    def correct_forward_lean(
        self,
        keypoints: np.ndarray,
        jcoords: Optional[np.ndarray] = None,
        angle: float | None = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Correct systematic forward lean. Auto-estimates angle if not given."""
        single_frame = keypoints.ndim == 2
        if single_frame:
            keypoints = keypoints[np.newaxis]
            if jcoords is not None:
                jcoords = jcoords[np.newaxis]

        if angle is None:
            angle = self._estimate_lean_angle(keypoints)

        if abs(angle) < 1.0:
            result_k = keypoints[0] if single_frame else keypoints
            if jcoords is not None:
                result_j = jcoords[0] if single_frame else jcoords
                return result_k, result_j
            return result_k

        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        corrected_k = keypoints.copy()
        corrected_j = jcoords.copy() if jcoords is not None else None
        for i in range(corrected_k.shape[0]):
            pelvis = (corrected_k[i, 9] + corrected_k[i, 10]) / 2
            corrected_k[i] = (corrected_k[i] - pelvis) @ rotation.T + pelvis
            if corrected_j is not None:
                corrected_j[i] = (corrected_j[i] - pelvis) @ rotation.T + pelvis

        result_k = corrected_k[0] if single_frame else corrected_k
        if corrected_j is not None:
            result_j = corrected_j[0] if single_frame else corrected_j
            return result_k, result_j
        return result_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_height_scale(
        self,
        kpts: np.ndarray,
        jc: Optional[np.ndarray],
    ) -> float:
        """
        Compute uniform scale so the skeleton matches self.subject_height.

        Top reference (after camera→OpenSim rotation, Y=up):
          - c_head from jcoords[113]  when jcoords are available
          - Nose  from kpts[0]        otherwise (divided by _NOSE_HEIGHT_FRACTION)

        Bottom reference: min Y among foot keypoints (heels + big toes).
        """
        heights = []
        N = kpts.shape[0]
        for i in range(N):
            foot_y = np.min(kpts[i, _FOOT_INDICES, 1])

            if jc is not None:
                top_y = jc[i, _JCOORDS_HEAD_IDX, 1]
            else:
                top_y = kpts[i, 0, 1] / self._NOSE_HEIGHT_FRACTION

            h = top_y - foot_y
            if h > 0.1:
                heights.append(h)

        if not heights:
            return 1.0
        return self.subject_height / float(np.mean(heights))

    def _pelvis_shifts(self, kpts: np.ndarray) -> np.ndarray:
        """Per-frame XZ shift to centre the pelvis; Y component is zero."""
        shifts = np.zeros((kpts.shape[0], 3), dtype=np.float64)
        for i in range(kpts.shape[0]):
            pelvis = (kpts[i, 9] + kpts[i, 10]) / 2
            shifts[i, 0] = pelvis[0]
            shifts[i, 2] = pelvis[2]
        return shifts

    def _align_to_ground(
        self, kpts: np.ndarray, return_offsets: bool = False
    ):
        result = kpts.copy()
        offsets = np.zeros(kpts.shape[0], dtype=np.float64)
        for i in range(kpts.shape[0]):
            min_y = np.min(kpts[i, _FOOT_INDICES, 1])
            result[i, :, 1] -= min_y
            offsets[i] = min_y
        if return_offsets:
            return result, offsets
        return result

    def _apply_global_translation(self, keypoints, camera_translation, scale):
        num_frames = keypoints.shape[0]
        if camera_translation.ndim == 1:
            camera_translation = np.tile(camera_translation, (num_frames, 1))
        cam_t_opensim = camera_translation @ self.CAMERA_TO_OPENSIM.T * scale
        cam_t_smoothed = self._smooth_cam_t(cam_t_opensim)
        first_frame_t = cam_t_smoothed[0].copy()
        xz_deltas = np.zeros((num_frames, 3))
        for i in range(num_frames):
            delta_t = cam_t_smoothed[i] - first_frame_t
            keypoints[i, :, 0] += delta_t[0]
            keypoints[i, :, 2] += delta_t[2]
            xz_deltas[i, 0] = delta_t[0]
            xz_deltas[i, 2] = delta_t[2]
        return keypoints, xz_deltas

    def _smooth_cam_t(self, cam_t, window_size=5):
        smoothed = cam_t.copy()
        for axis in range(3):
            smoothed[:, axis] = uniform_filter1d(cam_t[:, axis], size=window_size, mode="nearest")
        return smoothed

    def _estimate_lean_angle(self, keypoints: np.ndarray) -> float:
        """Estimate forward lean angle (degrees) from pelvis-thorax line vs vertical."""
        angles = []
        for i in range(keypoints.shape[0]):
            pelvis = (keypoints[i, 9] + keypoints[i, 10]) / 2
            thorax = (keypoints[i, 67] + keypoints[i, 68]) / 2
            spine_vec = thorax - pelvis
            xz = np.array([spine_vec[0], spine_vec[1]])
            if np.linalg.norm(xz) > 0.01:
                cos_a = np.dot(xz, [0, 1]) / np.linalg.norm(xz)
                a = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                angles.append(a if spine_vec[0] > 0 else -a)
        return float(np.median(angles)) if angles else 0.0
