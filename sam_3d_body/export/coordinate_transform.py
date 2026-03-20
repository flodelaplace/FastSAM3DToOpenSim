"""
Coordinate system transformation from camera to OpenSim.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter1d


class CoordinateTransformer:
    """
    Transforms coordinates from SAM3D Body camera space to OpenSim world space.

    Camera:  X=right, Y=down,    Z=forward(depth)
    OpenSim: X=forward(anterior), Y=up, Z=right(lateral)
    """

    CAMERA_TO_OPENSIM = np.array(
        [
            [0,  0, 1],   # X_opensim = Z_camera
            [0, -1, 0],   # Y_opensim = -Y_camera
            [1,  0, 0],   # Z_opensim = X_camera
        ],
        dtype=np.float64,
    )

    def __init__(self, subject_height: float = 1.75, units: str = "m"):
        self.subject_height = subject_height
        self.units = units
        self.scale_factor = 1000.0 if units == "mm" else 1.0

    def transform(
        self,
        keypoints_3d: np.ndarray,
        camera_translation: Optional[np.ndarray] = None,
        center_pelvis: bool = True,
        align_to_ground: bool = True,
        apply_global_translation: bool = False,
    ) -> np.ndarray:
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        transformed = keypoints_3d.copy().astype(np.float64)

        # Rotate axes: camera → OpenSim
        for i in range(transformed.shape[0]):
            transformed[i] = transformed[i] @ self.CAMERA_TO_OPENSIM.T

        # Scale to subject height
        transformed, height_scale = self._scale_to_subject(transformed, return_scale=True)

        if apply_global_translation and camera_translation is not None:
            transformed = self._apply_global_translation(transformed, camera_translation, height_scale)
        elif center_pelvis:
            transformed = self._center_at_pelvis(transformed)

        if align_to_ground:
            transformed = self._align_to_ground(transformed)

        transformed = transformed * self.scale_factor

        if single_frame:
            transformed = transformed[0]
        return transformed

    def _apply_global_translation(self, keypoints, camera_translation, height_scale):
        num_frames = keypoints.shape[0]
        if camera_translation.ndim == 1:
            camera_translation = np.tile(camera_translation, (num_frames, 1))
        cam_t_opensim = camera_translation @ self.CAMERA_TO_OPENSIM.T
        cam_t_opensim = cam_t_opensim * height_scale
        cam_t_smoothed = self._smooth_cam_t(cam_t_opensim)
        first_frame_t = cam_t_smoothed[0].copy()
        for i in range(num_frames):
            delta_t = cam_t_smoothed[i] - first_frame_t
            keypoints[i, :, 0] += delta_t[0]
            keypoints[i, :, 2] += delta_t[2]
        return keypoints

    def _smooth_cam_t(self, cam_t, window_size=5):
        smoothed = cam_t.copy()
        for axis in range(3):
            smoothed[:, axis] = uniform_filter1d(cam_t[:, axis], size=window_size, mode="nearest")
        return smoothed

    def _scale_to_subject(self, keypoints, return_scale=False):
        head_idx, left_ankle_idx, right_ankle_idx = 0, 13, 14
        heights = []
        for i in range(keypoints.shape[0]):
            head = keypoints[i, head_idx]
            ankle_mid = (keypoints[i, left_ankle_idx] + keypoints[i, right_ankle_idx]) / 2
            h = np.linalg.norm(head - ankle_mid)
            if h > 0.1:
                heights.append(h)
        scale = 1.0
        if heights:
            avg_height = np.mean(heights)
            estimated_full_height = avg_height * 1.1
            scale = self.subject_height / estimated_full_height
            keypoints = keypoints * scale
        if return_scale:
            return keypoints, scale
        return keypoints

    def _center_at_pelvis(self, keypoints):
        for i in range(keypoints.shape[0]):
            pelvis = (keypoints[i, 9] + keypoints[i, 10]) / 2
            keypoints[i, :, 0] -= pelvis[0]
            keypoints[i, :, 2] -= pelvis[2]
        return keypoints

    def _align_to_ground(self, keypoints):
        for i in range(keypoints.shape[0]):
            foot_heights = [
                keypoints[i, 17, 1],  # left_heel
                keypoints[i, 20, 1],  # right_heel
                keypoints[i, 15, 1],  # left_big_toe
                keypoints[i, 18, 1],  # right_big_toe
            ]
            min_y = min(foot_heights)
            keypoints[i, :, 1] -= min_y
        return keypoints

    def correct_forward_lean(self, keypoints: np.ndarray, angle: float | None = None) -> np.ndarray:
        """Correct systematic forward lean. Auto-estimates angle if not given."""
        single_frame = keypoints.ndim == 2
        if single_frame:
            keypoints = keypoints[np.newaxis, ...]

        if angle is None:
            angle = self._estimate_lean_angle(keypoints)

        if abs(angle) < 1.0:
            return keypoints[0] if single_frame else keypoints

        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        corrected = keypoints.copy()
        for i in range(corrected.shape[0]):
            pelvis = (corrected[i, 9] + corrected[i, 10]) / 2
            corrected[i] = corrected[i] - pelvis
            corrected[i] = corrected[i] @ rotation.T
            corrected[i] = corrected[i] + pelvis

        return corrected[0] if single_frame else corrected

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
