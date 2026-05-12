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
        correct_floor_lean: bool = True,
        floor_angle: Optional[float] = None,
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
        self._last_scale = scale
        kpts = kpts * scale
        if jc is not None:
            jc = jc * scale

        # 3. Global translation / pelvis centering
        # NB: on capture les deltas (xz_deltas ou shift) en mètres pour pouvoir
        # rejouer le même décalage sur d'autres arrays (mesh verts) via
        # apply_pipeline_to_verts() — utile pour aligner le mesh GLB avec
        # l'anatomical GLB qui passe par la même pipeline kpts.
        self._last_xz_deltas_m = None
        self._last_pelvis_shifts_m = None
        if apply_global_translation and camera_translation is not None:
            kpts, xz_deltas = self._apply_global_translation(kpts, camera_translation, scale)
            if jc is not None:
                jc[:, :, 0] += xz_deltas[:, 0:1]
                jc[:, :, 2] += xz_deltas[:, 2:3]
            self._last_xz_deltas_m = xz_deltas.copy()  # (N, 3) in meters
        elif center_pelvis:
            shift = self._pelvis_shifts(kpts)      # (N, 3) with Y=0
            kpts = kpts - shift[:, None, :]
            if jc is not None:
                jc = jc - shift[:, None, :]
            self._last_pelvis_shifts = shift * self.scale_factor  # saved in output units (mm)
            self._last_pelvis_shifts_m = shift.copy()  # (N, 3) in meters

        # 3b. Floor-plane lean correction — must run BEFORE per-frame align_to_ground,
        #     which destroys the global floor-tilt signal by independently shifting
        #     every frame.  Fit a line to stance-foot positions across all frames;
        #     the slope reveals the camera pitch → rotate the skeleton to level the floor.
        self._last_floor_angle_deg = None
        if align_to_ground and correct_floor_lean:
            if floor_angle is not None:
                _angle = floor_angle
                _src = "MoGe"
            else:
                _angle = self._fit_floor_plane_angle(kpts)
                _src = "foot-traj"
            if abs(_angle) > 0.5:
                print(f"  [floor lean] {_src} floor tilt {_angle:+.2f}° → correcting")
                kpts, jc = self._rotate_around_pelvis_z(kpts, jc, _angle)
                self._last_floor_angle_deg = _angle

        # 4. Align feet to Y=0 each frame
        self._last_ground_offsets_m = None
        if align_to_ground:
            kpts, ground_offsets = self._align_to_ground(kpts, return_offsets=True)
            if jc is not None:
                jc[:, :, 1] -= ground_offsets[:, None]
            self._last_ground_offsets_m = ground_offsets.copy()  # (N,) in meters

        # 5. Unit conversion (m → mm if requested)
        kpts = kpts * self.scale_factor
        if jc is not None:
            jc = jc * self.scale_factor

        if single_frame:
            kpts = kpts[0]
            if jc is not None:
                jc = jc[0]

        return (kpts, jc) if jcoords_3d is not None else kpts

    def apply_pipeline_to_verts(
        self,
        verts_per_frame: list,
        output_units: str = "m",
        ground_offset_mode: str = "per_frame",
        calib_window_frames: int = 20,
        override_constant_offset_m: float | None = None,
    ) -> list:
        """Rejoue le pipeline transform() sur des points 3D arbitraires
        (typiquement des vertices de mesh) en utilisant l'état capturé par
        le dernier appel à transform(). Permet d'aligner le mesh GLB sur le
        même repère world OpenSim que les keypoints / anatomical GLB.

        Args:
            verts_per_frame : liste de N arrays (M_i, 3) ou None.
                              Points en CAMERA-WORLD frame (mesh_local + cam_t).
            output_units : "m" ou "mm".
            ground_offset_mode :
                "per_frame"          → applique ground_offsets[i] par frame (comme kpts).
                                        Inconvénient pour le mesh : les vertices peuvent
                                        descendre SOUS Y=0 si le mesh's lowest est sous
                                        les kpts feet markers.
                "constant_from_calib" → calcule UN shift Y unique depuis le mesh lui-même
                                        sur les `calib_window_frames` premières frames
                                        (fenêtre standing), puis l'applique constant.
                                        Plus naturel : le mesh est au sol debout, et
                                        flotte normalement quand les pieds montent.
                "none"                → pas de ground alignment (utile pour debug).
            calib_window_frames : nb de frames pour calculer l'offset constant.

        Returns:
            Liste de N arrays (M_i, 3) ou None, dans le frame OpenSim world.

        Note: l'appel à transform() doit avoir été fait juste avant pour
        que l'état soit cohérent. La méthode est read-only (ne modifie
        pas l'état du transformer).
        """
        if self._last_scale is None:
            raise RuntimeError(
                "Call transform() first to populate transformation state.")

        out_scale = 1000.0 if output_units == "mm" else 1.0

        # Étapes 1-4 (rotation, scale, XZ shifts, floor lean) — appliquées
        # per-frame comme pour les keypoints.
        pre_ground: list = []
        for i, v in enumerate(verts_per_frame):
            if v is None:
                pre_ground.append(None)
                continue
            w = np.asarray(v, dtype=np.float64).copy()
            w = w @ self.CAMERA_TO_OPENSIM.T
            w = w * self._last_scale
            if self._last_xz_deltas_m is not None and i < len(self._last_xz_deltas_m):
                d = self._last_xz_deltas_m[i]
                w[:, 0] += d[0]
                w[:, 2] += d[2]
            elif self._last_pelvis_shifts_m is not None and i < len(self._last_pelvis_shifts_m):
                shift = self._last_pelvis_shifts_m[i]
                w -= shift[None, :]
            if (self._last_floor_angle_deg is not None
                    and abs(self._last_floor_angle_deg) > 0.5
                    and hasattr(self, "_last_floor_pivots_m")
                    and i < len(self._last_floor_pivots_m)):
                theta = np.radians(self._last_floor_angle_deg)
                c, s = np.cos(theta), np.sin(theta)
                Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
                pivot = self._last_floor_pivots_m[i]
                w = (w - pivot) @ Rz + pivot
            pre_ground.append(w)

        # Étape 5 — Ground alignment (mode-dependent)
        if ground_offset_mode == "per_frame":
            # Réutilise les offsets calculés sur kpts. Souci possible : si le
            # mesh a des vertices plus bas que les feet markers (toes, heels),
            # le mesh descendra sous Y=0.
            for i, w in enumerate(pre_ground):
                if w is None or self._last_ground_offsets_m is None:
                    continue
                if i < len(self._last_ground_offsets_m):
                    w[:, 1] -= self._last_ground_offsets_m[i]
        elif ground_offset_mode == "constant_from_calib":
            # Si override_constant_offset_m est fourni → on l'applique tel quel
            # (utilisé pour partager le MÊME offset entre plusieurs arrays
            # mesh/kpts/jcoords afin qu'ils restent alignés entre eux dans le
            # GLB final). Sinon, on calcule l'offset depuis les premières
            # frames de l'array passé (calib window standing typique).
            if override_constant_offset_m is not None:
                constant_offset = float(override_constant_offset_m)
            else:
                calib_ys = []
                for i, w in enumerate(pre_ground[:calib_window_frames]):
                    if w is not None:
                        calib_ys.append(float(w[:, 1].min()))
                if not calib_ys:
                    constant_offset = 0.0
                else:
                    constant_offset = float(min(calib_ys))
                    print(f"  [apply_pipeline_to_verts] constant_from_calib: "
                          f"Y -= {constant_offset:.3f} m (calib over "
                          f"{len(calib_ys)} frames)")
            for w in pre_ground:
                if w is not None:
                    w[:, 1] -= constant_offset
        # ground_offset_mode == "none" : pas de shift Y

        # Étape 6 — Unit conversion + float32
        out: list = []
        for w in pre_ground:
            if w is None:
                out.append(None)
            else:
                out.append((w * out_scale).astype(np.float32))
        return out

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

        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        corrected_k = keypoints.copy()
        corrected_j = jcoords.copy() if jcoords is not None else None
        for i in range(corrected_k.shape[0]):
            pelvis = (corrected_k[i, 9] + corrected_k[i, 10]) / 2
            corrected_k[i] = (corrected_k[i] - pelvis) @ rotation.T + pelvis
            if corrected_j is not None:
                corrected_j[i] = (corrected_j[i] - pelvis) @ rotation.T + pelvis

        # Re-align feet to ground after rotation — without this the person
        # floats above or sinks below Y=0.
        corrected_k, ground_offsets = self._align_to_ground(corrected_k, return_offsets=True)
        if corrected_j is not None:
            corrected_j[:, :, 1] -= ground_offsets[:, None]

        result_k = corrected_k[0] if single_frame else corrected_k
        if corrected_j is not None:
            result_j = corrected_j[0] if single_frame else corrected_j
            return result_k, result_j
        return result_k

    def correct_lean_cam_pitch(
        self,
        keypoints: np.ndarray,
        jcoords: Optional[np.ndarray] = None,
        cam_t: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Correct forward lean by estimating camera pitch from the cam_t trajectory.

        The body root walks through camera space; any systematic drift of
        cam_t_opensim.Y with cam_t_opensim.X reveals the camera tilt angle θ:
            d(Y_opensim)/d(X_opensim) = -tan(θ)  →  θ = -arctan(slope)

        A Rz(θ) rotation (around OpenSim Z = lateral axis) is applied to every
        frame, pivoting around the pelvis, followed by ground re-alignment.

        keypoints/jcoords must already be in OpenSim space (metres or mm).
        cam_t must be in camera space (raw, before rotation/scaling).
        """
        if cam_t is None or len(cam_t) < 4:
            result_k = keypoints
            if jcoords is not None:
                return result_k, jcoords
            return result_k

        angle = self._estimate_pitch_angle(cam_t)
        if abs(angle) < 0.5:
            if jcoords is not None:
                return keypoints, jcoords
            return keypoints

        single_frame = keypoints.ndim == 2
        if single_frame:
            keypoints = keypoints[np.newaxis]
            if jcoords is not None:
                jcoords = jcoords[np.newaxis]

        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        # Rotation around OpenSim Z (lateral): tilts X↔Y
        Rz = np.array([
            [ cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [     0,     0, 1],
        ], dtype=np.float64)

        corrected_k = keypoints.copy()
        corrected_j = jcoords.copy() if jcoords is not None else None
        for i in range(corrected_k.shape[0]):
            pelvis = (corrected_k[i, 9] + corrected_k[i, 10]) / 2
            corrected_k[i] = (corrected_k[i] - pelvis) @ Rz.T + pelvis
            if corrected_j is not None:
                corrected_j[i] = (corrected_j[i] - pelvis) @ Rz.T + pelvis

        # Re-align feet to ground after rotation
        corrected_k, ground_offsets = self._align_to_ground(corrected_k, return_offsets=True)
        if corrected_j is not None:
            corrected_j[:, :, 1] -= ground_offsets[:, None]

        if single_frame:
            corrected_k = corrected_k[0]
            if corrected_j is not None:
                corrected_j = corrected_j[0]

        if corrected_j is not None:
            return corrected_k, corrected_j
        return corrected_k

    def _estimate_pitch_angle(self, cam_t: np.ndarray) -> float:
        """
        Estimate camera pitch angle (degrees) from the cam_t trajectory.

        cam_t is in camera space (raw, shape (N,3)). We convert to OpenSim
        axes and fit a line to Y_opensim vs X_opensim.  The slope equals
        -tan(θ), so θ = -arctan(slope).  A positive θ means the camera
        points downward, which makes the body appear to lean forward.
        """
        scale = getattr(self, "_last_scale", 1.0)
        ct_opensim = cam_t @ self.CAMERA_TO_OPENSIM.T * scale

        x = ct_opensim[:, 0]  # forward/depth in OpenSim
        y = ct_opensim[:, 1]  # up

        # Need enough horizontal travel to get a meaningful slope
        x_range = np.ptp(x)
        if x_range < 0.05:
            return 0.0

        slope, _ = np.polyfit(x, y, 1)
        angle = -np.degrees(np.arctan(slope))
        return float(angle)

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

    def _fit_floor_plane_angle(self, kpts: np.ndarray) -> float:
        """
        Estimate floor tilt in the sagittal plane (rotation around OpenSim Z / lateral axis).

        For each frame, takes the minimum-Y foot among _FOOT_INDICES (the stance foot),
        then fits Y = a*X + b to those (forward, height) pairs across all frames.
        Slope a > 0 means the floor appears to rise going forward (camera pitched down),
        which makes the body look like it leans forward.

        Works whether keypoints are pelvis-centred (pelvis-relative positions) or
        have global translation applied (full walking trajectory).  For photos / static
        poses where all frames are identical, the X range is zero and the function
        returns 0.0 (no correction).

        Parameters
        ----------
        kpts : (N, 70, 3) in OpenSim space, scaled to metres, NOT yet ground-aligned.

        Returns
        -------
        float : tilt angle in degrees.  Positive → floor rises going forward →
                correct by rotating the skeleton backward (Rz applied inside
                _rotate_around_pelvis_z).  Clamped to ±20°.
        """
        pts = []
        for i in range(kpts.shape[0]):
            foot = kpts[i, _FOOT_INDICES]       # (4, 3)
            if np.any(np.isnan(foot)):
                continue
            pts.append(foot[np.argmin(foot[:, 1])])   # stance foot = lowest
        if len(pts) < 4:
            return 0.0
        pts = np.array(pts)                     # (M, 3)
        x_range = np.ptp(pts[:, 0])
        if x_range < 0.05:                      # < 5 cm forward travel — not enough signal
            return 0.0
        try:
            slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
        except np.linalg.LinAlgError:
            return 0.0
        angle = float(np.degrees(np.arctan(slope)))
        return float(np.clip(angle, -20.0, 20.0))

    @staticmethod
    def floor_angle_from_moge_points(
        points: np.ndarray,
        mask: np.ndarray,
        person_bbox=None,
        orig_hw=None,
        floor_frac: float = 0.25,
        n_samples: int = 4000,
    ) -> float:
        """
        Estimate the lean correction angle from MoGe 3D points on the floor.

        MoGe uses a Y-UP camera convention (X=right, Y=up, Z=forward).
        Floor points are BELOW the optical axis → most negative Y_cam/Z_cam.

        Algorithm
        ---------
        1. Keep only bottom floor_frac of image rows (by Y/Z image-space coord).
        2. Exclude person bounding box pixels (person_bbox in original frame coords).
        3. SVD plane fit → floor normal n̂ → camera pitch θ = arctan(n_z / n_y).
        4. Scale raw camera pitch by LEAN_SCALE.

        Why LEAN_SCALE < 1:
        MoGe measures the CAMERA PITCH accurately, but the body pose model (MHR)
        already compensates for ~68% of the camera pitch internally.  Only the
        remaining ~32% manifests as a skeleton lean artifact that needs correcting.
        Additionally, OpenSim IK amplifies keypoint lean by ~2× in the visual output,
        so the effective correction scale is approximately 0.32 / 2 ≈ 0.33.
        This was calibrated empirically on aitor_garden_walk.mp4 (camera pitch ≈ 31°,
        needed keypoint rotation ≈ 10°, scale = 10/31 ≈ 0.32).

        Parameters
        ----------
        person_bbox : (x1, y1, x2, y2) in original frame pixel coords — excluded
        orig_hw     : (H, W) of original frame — needed to scale bbox to MoGe grid
        floor_frac  : fraction of image rows from the bottom to consider as floor

        Returns lean correction angle in degrees. Clamped to ±15°.
        """
        LEAN_SCALE = 0.33  # camera-pitch → lean-correction scale (see docstring)
        # points must be (H, W, 3) spatial grid
        if points.ndim != 3:
            return 0.0
        H, W = points.shape[:2]

        # Build per-pixel valid mask (H, W)
        valid_mask = mask.astype(bool)  # (H, W)

        # Exclude person bounding box pixels
        if person_bbox is not None and orig_hw is not None:
            oh, ow = orig_hw
            x1, y1, x2, y2 = person_bbox
            # Scale bbox from original frame to MoGe grid
            gx1 = int(x1 / ow * W)
            gy1 = int(y1 / oh * H)
            gx2 = int(x2 / ow * W)
            gy2 = int(y2 / oh * H)
            # Add margin: expand bbox by 10% on each side
            margin_x = max(1, int((gx2 - gx1) * 0.10))
            margin_y = max(1, int((gy2 - gy1) * 0.10))
            gx1 = max(0, gx1 - margin_x)
            gy1 = max(0, gy1 - margin_y)
            gx2 = min(W - 1, gx2 + margin_x)
            gy2 = min(H - 1, gy2 + margin_y)
            valid_mask[gy1:gy2+1, gx1:gx2+1] = False

        # Flatten to valid points
        pts_flat = points.reshape(-1, 3).astype(np.float64)
        mask_flat = valid_mask.reshape(-1)
        valid = pts_flat[mask_flat]
        if len(valid) < 50:
            return 0.0

        # Floor candidates: depth-normalized image-row position Y/Z = -(v-cy)/fy
        # Depth-independent — bottom floor_frac of image rows regardless of distance
        y_norm = valid[:, 1] / valid[:, 2]  # Y_cam / Z_cam
        thresh = np.percentile(y_norm, floor_frac * 100)
        floor_pts = valid[y_norm <= thresh]

        if len(floor_pts) < 20:
            return 0.0

        # Random subsample for speed
        if len(floor_pts) > n_samples:
            idx = np.random.choice(len(floor_pts), n_samples, replace=False)
            floor_pts = floor_pts[idx]

        # SVD plane fit in camera space
        centroid = floor_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(floor_pts - centroid, full_matrices=False)
        normal = Vt[-1]   # smallest singular value → plane normal

        # Ensure normal points upward in camera space (positive Y_cam)
        if normal[1] < 0:
            normal = -normal

        # Camera pitch: angle between floor normal and Y_cam axis
        # θ = arctan(n_z / n_y): positive when n_z > 0 (camera tilts down)
        raw_angle = float(np.degrees(np.arctan2(normal[2], normal[1])))
        correction = raw_angle * LEAN_SCALE
        print(f"  [floor_moge] floor candidates: {len(floor_pts)}, "
              f"Z_mean={floor_pts[:,2].mean():.1f}, "
              f"normal=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}], "
              f"camera_pitch={raw_angle:+.2f}°, correction={correction:+.2f}°")
        return float(np.clip(correction, -15.0, 15.0))

    def _rotate_around_pelvis_z(
        self,
        kpts: np.ndarray,
        jc,
        angle_deg: float,
    ) -> tuple:
        """
        Rotate the skeleton around the per-frame pelvis pivot by angle_deg around
        the OpenSim Z (lateral) axis.

        Row-vector convention: applies Rz(angle_deg) as  v @ Rz  (no transpose).
        Positive angle_deg tilts the body backward, correcting forward lean.

        Derivation: for a point (x, y) on the floor at height y = slope * x,
        rotation by arctan(slope) maps y → 0 (floor becomes horizontal).
        """
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        pivots = np.zeros((kpts.shape[0], 3), dtype=np.float64)
        for i in range(kpts.shape[0]):
            pelvis = (kpts[i, 9] + kpts[i, 10]) / 2
            pivots[i] = pelvis
            kpts[i] = (kpts[i] - pelvis) @ Rz + pelvis
            if jc is not None:
                jc[i] = (jc[i] - pelvis) @ Rz + pelvis
        # Capture pivots so apply_pipeline_to_verts() peut refaire la même
        # rotation autour du même axe pour des arrays externes (mesh verts).
        self._last_floor_pivots_m = pivots
        return kpts, jc

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