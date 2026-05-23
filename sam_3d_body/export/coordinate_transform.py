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
        self._last_stationary_cam_t_m = None
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
            # Stationary mode : transform() ignore cam_t XZ pour les kpts (pelvis
            # centré XZ), mais on ajoute cam_t_Y pour préserver la motion
            # verticale naturelle (pelvis qui descend pendant un squat). Sinon
            # l'anatomical reste bloqué à pelvis_Y = 0 (canonical SMPL).
            #
            # Le mesh (apply_pipeline_to_verts) reçoit verts + cam_t baked déjà,
            # on stocke cam_t_XZ (Y=0) pour soustraire seulement le XZ et garder
            # le Y. Comme ça mesh et kpts sont symétriques.
            if camera_translation is not None:
                ct_os = (camera_translation @ self.CAMERA_TO_OPENSIM.T * scale
                         ).astype(np.float64)  # (N, 3) in meters
                # Inject cam_t_Y aux kpts pour préserver vertical motion
                kpts[:, :, 1] += ct_os[:, 1:2]
                if jc is not None:
                    jc[:, :, 1] += ct_os[:, 1:2]
                # Stocker XZ-only pour soustraction côté mesh (Y=0 préservé)
                ct_os_xz = ct_os.copy()
                ct_os_xz[:, 1] = 0
                self._last_stationary_cam_t_m = ct_os_xz

        # 3b. Floor-plane lean correction — must run BEFORE per-frame align_to_ground,
        #     which destroys the global floor-tilt signal by independently shifting
        #     every frame.  Fit a line to stance-foot positions across all frames;
        #     the slope reveals the camera pitch → rotate the skeleton to level the floor.
        self._last_floor_angle_deg = None
        self._last_roll_angle_deg = None
        self._last_body_vertical_R = None
        if correct_floor_lean:
            if floor_angle is not None:
                # MoGe : tuple (pitch, roll). Back-compat float = pitch seul.
                if isinstance(floor_angle, tuple):
                    _pitch, _roll = floor_angle
                else:
                    _pitch, _roll = float(floor_angle), 0.0
                _src = "MoGe"
            elif align_to_ground:
                _pitch = self._fit_floor_plane_angle(kpts)
                _roll = 0.0
                _src = "foot-traj"
            else:
                _pitch, _roll, _src = 0.0, 0.0, None
            if _src is not None and abs(_pitch) > 0.5:
                print(f"  [floor lean] {_src} pitch {_pitch:+.2f}° → correcting")
                kpts, jc = self._rotate_around_pelvis_z(kpts, jc, _pitch)
                self._last_floor_angle_deg = _pitch
            if _src is not None and abs(_roll) > 0.5:
                print(f"  [floor lean] {_src} roll  {_roll:+.2f}° → correcting")
                kpts, jc = self._rotate_around_pelvis_x(kpts, jc, _roll)
                self._last_roll_angle_deg = _roll

            # Body-vertical correction : SEULEMENT si --floor (= align_to_ground)
            # car ça utilise la posture du sujet (assume standing) qui n'est
            # pas valide pour rameur/LASEGUE/suspendu. Le mode défaut se base
            # uniquement sur MoGe (= signal du sol).
            if align_to_ground:
                kpts, jc = self._apply_body_vertical_correction(kpts, jc)

        # 4. Align feet to Y=0
        self._last_ground_offsets_m = None
        if align_to_ground:
            # --floor : per-frame ground align (feet à Y=0 chaque frame)
            kpts, ground_offsets = self._align_to_ground(kpts, return_offsets=True)
            if jc is not None:
                jc[:, :, 1] -= ground_offsets[:, None]
            self._last_ground_offsets_m = ground_offsets.copy()  # (N,) in meters
        elif correct_floor_lean:
            # Mode défaut (MoGe sans --floor) : shift constant calibré sur les
            # 20 premières frames pour mettre les pieds proche du sol initial.
            # Sans ça l'anatomical apparaît sous le sol (kpts.Y = cam_t_y brut).
            # Le shift est appliqué à TOUTES les frames de manière constante,
            # ce qui préserve la motion Y naturelle (squat, sauts, etc).
            n_calib = min(20, kpts.shape[0])
            calib_min_y = []
            for i in range(n_calib):
                foot = kpts[i, _FOOT_INDICES]
                if not np.any(np.isnan(foot)):
                    calib_min_y.append(np.min(foot[:, 1]))
            if calib_min_y:
                constant_offset = float(np.min(calib_min_y))
                print(f"  [floor lean] constant ground shift Y -= {constant_offset:.3f} m "
                      f"(calib over {len(calib_min_y)} frames)")
                kpts[:, :, 1] -= constant_offset
                if jc is not None:
                    jc[:, :, 1] -= constant_offset

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
                # Stationary mode : retire aussi cam_t (rotated+scaled) car les
                # mesh verts l'avaient baked in lors de l'append (verts+cam_t).
                # Sans ça, mesh = pelvis_local-centered + cam_t résiduel → wobble.
                if (self._last_stationary_cam_t_m is not None
                        and i < len(self._last_stationary_cam_t_m)):
                    w -= self._last_stationary_cam_t_m[i][None, :]
            # Pitch correction (axe Z lateral)
            if (self._last_floor_angle_deg is not None
                    and abs(self._last_floor_angle_deg) > 0.5
                    and hasattr(self, "_last_floor_pivots_m")
                    and i < len(self._last_floor_pivots_m)):
                theta = np.radians(self._last_floor_angle_deg)
                c, s = np.cos(theta), np.sin(theta)
                Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
                pivot = self._last_floor_pivots_m[i]
                w = (w - pivot) @ Rz + pivot
            # Roll correction (axe X anterior), APRÈS pitch
            if (getattr(self, "_last_roll_angle_deg", None) is not None
                    and abs(self._last_roll_angle_deg) > 0.5
                    and hasattr(self, "_last_roll_pivots_m")
                    and i < len(self._last_roll_pivots_m)):
                theta = np.radians(self._last_roll_angle_deg)
                c, s = np.cos(theta), np.sin(theta)
                Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
                pivot = self._last_roll_pivots_m[i]
                w = (w - pivot) @ Rx + pivot
            # Body-vertical correction (3D Rodrigues from standing window)
            if (getattr(self, "_last_body_vertical_R", None) is not None
                    and hasattr(self, "_last_body_vertical_pivots_m")
                    and i < len(self._last_body_vertical_pivots_m)):
                R_bv = self._last_body_vertical_R
                pivot = self._last_body_vertical_pivots_m[i]
                w = (w - pivot) @ R_bv.T + pivot
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
    ) -> tuple:
        """
        Estimate floor lean angles (pitch, roll) from MoGe 3D points.
        Returns (pitch_deg, roll_deg) tuple — both clamped to ±60°.

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

        Returns (pitch_deg, roll_deg) tuple - both clamped to ±60°.
        """
        # LEAN_SCALE = 0.5 → compromis entre 0.33 (calibré pour aitor_garden_walk,
        # sous-correction sur autres vidéos) et 1.0 (sur-correction si MoGe
        # surestime le pitch caméra). Peut être ajusté par vidéo via env
        # variable MOGE_LEAN_SCALE pour debug.
        import os as _os
        LEAN_SCALE = float(_os.environ.get("MOGE_LEAN_SCALE", "0.5"))
        # points must be (H, W, 3) spatial grid
        if points.ndim != 3:
            return (0.0, 0.0)
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
            return (0.0, 0.0)

        # Floor candidates: depth-normalized image-row position Y/Z = -(v-cy)/fy
        # Depth-independent — bottom floor_frac of image rows regardless of distance
        y_norm = valid[:, 1] / valid[:, 2]  # Y_cam / Z_cam
        thresh = np.percentile(y_norm, floor_frac * 100)
        floor_pts = valid[y_norm <= thresh]

        if len(floor_pts) < 20:
            return (0.0, 0.0)

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

        # Camera pitch: angle dans plan YZ (rotation autour axe X cam = lateral)
        # θ_pitch = arctan(n_z / n_y): positive when n_z > 0 (caméra penche bas)
        raw_pitch = float(np.degrees(np.arctan2(normal[2], normal[1])))
        # Camera roll: angle dans plan XY (rotation autour axe Z cam = forward)
        # θ_roll = arctan(n_x / n_y): positive when n_x > 0 (caméra penche droite)
        raw_roll = float(np.degrees(np.arctan2(normal[0], normal[1])))
        correction_pitch = raw_pitch * LEAN_SCALE
        correction_roll = raw_roll * LEAN_SCALE
        print(f"  [floor_moge] floor candidates: {len(floor_pts)}, "
              f"Z_mean={floor_pts[:,2].mean():.1f}, "
              f"normal=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}], "
              f"camera_pitch={raw_pitch:+.2f}° → {correction_pitch:+.2f}°, "
              f"camera_roll={raw_roll:+.2f}° → {correction_roll:+.2f}°")
        # Roll négé pour matcher la convention de _rotate_around_pelvis_x.
        # Env var MOGE_DISABLE_ROLL=1 pour tester sans roll si suspicion.
        DISABLE_ROLL = bool(int(_os.environ.get("MOGE_DISABLE_ROLL", "0")))
        return (
            float(np.clip(correction_pitch, -60.0, 60.0)),
            0.0 if DISABLE_ROLL else float(np.clip(-correction_roll, -60.0, 60.0)),
        )

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

    def _apply_body_vertical_correction(self, kpts, jc, calib_n_frames=20,
                                          min_cos_threshold=0.5):
        """Rotate so that the subject's body axis (midfoot→neck) aligns with +Y.
        Uses first calib_n_frames as standing reference. Skips if body axis
        isn't already mostly vertical (cos(angle_with_Y) < min_cos_threshold)
        — protects against non-standing subjects (LASEGUE etc.)."""
        n = min(calib_n_frames, kpts.shape[0])
        midfeet = []
        necks = []
        for i in range(n):
            foot = kpts[i, _FOOT_INDICES]
            if np.any(np.isnan(foot)):
                continue
            midfoot = np.mean(foot, axis=0)
            # neck = midpoint of left-shoulder (5) and right-shoulder (6)
            neck = (kpts[i, 5] + kpts[i, 6]) / 2
            if np.any(np.isnan(neck)):
                continue
            midfeet.append(midfoot)
            necks.append(neck)
        if len(midfeet) < 5:
            return kpts, jc
        mean_midfoot = np.mean(midfeet, axis=0)
        mean_neck = np.mean(necks, axis=0)
        body_axis = mean_neck - mean_midfoot
        norm = np.linalg.norm(body_axis)
        if norm < 1e-6:
            return kpts, jc
        body_axis = body_axis / norm
        target = np.array([0.0, 1.0, 0.0])
        cos_a = float(np.dot(body_axis, target))
        if cos_a < min_cos_threshold:
            print(f"  [body-vertical] body axis Y={cos_a:.2f} < threshold "
                  f"{min_cos_threshold} (subject probably not standing), skip")
            return kpts, jc
        axis = np.cross(body_axis, target)
        sin_a = float(np.linalg.norm(axis))
        if sin_a < 1e-6:
            return kpts, jc
        axis = axis / sin_a
        angle = float(np.degrees(np.arctan2(sin_a, cos_a)))
        if abs(angle) < 0.5:
            return kpts, jc
        print(f"  [body-vertical] body axis tilt {angle:.2f}° → correcting "
              f"(axis=[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}])")
        # Rodrigues rotation matrix (column-vector form)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        R = np.eye(3) + sin_a * K + (1.0 - cos_a) * K @ K
        # Apply to all frames around per-frame pelvis pivot
        pivots = np.zeros((kpts.shape[0], 3), dtype=np.float64)
        for i in range(kpts.shape[0]):
            pelvis = (kpts[i, 9] + kpts[i, 10]) / 2
            pivots[i] = pelvis
            # Row vector convention : v @ R.T applique R (column form) sur v
            kpts[i] = (kpts[i] - pelvis) @ R.T + pelvis
            if jc is not None:
                jc[i] = (jc[i] - pelvis) @ R.T + pelvis
        self._last_body_vertical_R = R
        self._last_body_vertical_pivots_m = pivots
        return kpts, jc

    def _rotate_around_pelvis_x(self, kpts, jc, angle_deg):
        """Rotate around per-frame pelvis pivot autour de l'axe X (anterior).
        Utilisé pour corriger le roll caméra (caméra penchée sur le côté)."""
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        pivots = np.zeros((kpts.shape[0], 3), dtype=np.float64)
        for i in range(kpts.shape[0]):
            pelvis = (kpts[i, 9] + kpts[i, 10]) / 2
            pivots[i] = pelvis
            kpts[i] = (kpts[i] - pelvis) @ Rx + pelvis
            if jc is not None:
                jc[i] = (jc[i] - pelvis) @ Rx + pelvis
        self._last_roll_pivots_m = pivots
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