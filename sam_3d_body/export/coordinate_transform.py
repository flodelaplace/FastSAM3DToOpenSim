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
        apply_body_vertical: Optional[bool] = None,
        lock_vertical: bool = False,
        lock_lateral: bool = False,
        contact_anchor: bool = False,
        fps: float = 30.0,
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
        self._last_stationary_cam_t_y_m = None
        if apply_global_translation and camera_translation is not None:
            kpts, xz_deltas = self._apply_global_translation(
                kpts, camera_translation, scale, lock_lateral=lock_lateral)
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
            # Le mesh (apply_pipeline_to_verts) reçoit verts RAW (sans cam_t
            # ajouté), donc on stocke cam_t.Y pour qu'il puisse ré-appliquer
            # exactement la même injection verticale que les kpts (et ainsi
            # rester aligné avec l'anatomical, qui lui suit kpts via TRC/IK).
            if camera_translation is not None and not lock_vertical:
                ct_os = (camera_translation @ self.CAMERA_TO_OPENSIM.T * scale
                         ).astype(np.float64)  # (N, 3) in meters
                # Inject cam_t_Y aux kpts pour préserver vertical motion
                kpts[:, :, 1] += ct_os[:, 1:2]
                if jc is not None:
                    jc[:, :, 1] += ct_os[:, 1:2]
                # Stocker la Y-injection pour la rejouer côté mesh.
                self._last_stationary_cam_t_y_m = ct_os[:, 1].copy()
            # lock_vertical=True → on skip complètement l'injection cam_t_Y.
            # Résultat : pelvis reste à Y=0 canonical (invariant en Y frame par
            # frame). Comme apply_pipeline_to_verts consulte le même
            # `_last_stationary_cam_t_y_m` (qui reste None), le mesh ne reçoit
            # PAS d'injection Y non plus → mesh + anatomical restent alignés
            # ET tous deux verrouillés verticalement. Idéal pour rendu bikefit
            # indoor / home-trainer où on ne veut pas l'oscillation du pédalage.

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

            # Body-vertical correction : utilise la posture du sujet pour
            # forcer midfoot→neck à être vertical. Suppose le sujet DEBOUT.
            # → Ne pas activer pour rameur, sit-to-stand, Lasègue, suspension.
            # Default = on si --floor (legacy : align_to_ground True).
            # `apply_body_vertical` permet l'override explicite (ex : --floor
            # avec sujet assis sur chaise = pieds au sol mais corps incliné).
            _do_body_vertical = (
                apply_body_vertical if apply_body_vertical is not None
                else align_to_ground
            )
            if _do_body_vertical:
                kpts, jc = self._apply_body_vertical_correction(kpts, jc)

        # 4. Align feet to Y=0
        self._last_ground_offsets_m = None
        self._last_constant_offset_m = None
        self._last_penetration_clamp_m = None  # (N,) per-frame safety-net shift
        if contact_anchor:
            # Ancrage conscient du contact : re-ancre le pied en contact SOUTENU
            # (squat → bassin descend) mais tient l'offset pendant un vol BREF
            # (course/saut → pas de "saut"). Unifie --floor (per-frame) et le
            # mode défaut. Stocke un offset per-frame (comme align_to_ground) →
            # le mesh GLB le rejoue via _last_ground_offsets_m.
            kpts, ground_offsets = self._contact_aware_ground(
                kpts, fps=fps, return_offsets=True)
            if jc is not None:
                jc[:, :, 1] -= ground_offsets[:, None]
            self._last_ground_offsets_m = ground_offsets.copy()
            # NB : l'anti-glisse pied ne se fait PAS ici (les kpts MHR ne sont
            # pas les marqueurs finaux). Elle s'applique sur les marqueurs
            # Flodelaplace du TRC via anti_foot_skate_markers() (voir pipeline).
        elif align_to_ground:
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
                # Store for exact replay in apply_pipeline_to_verts (alignement
                # mesh vs anatomical à zéro écart).
                self._last_constant_offset_m = constant_offset

            # 4b. Ground-penetration one-directional clamp (fix STS + bug MHR
            # sur mouvements assis). Per-frame safety : si min_foot_Y < 0
            # (pieds sous le sol) → shift UP pour min_foot_Y = 0. Ne shift
            # PAS DOWN si feet > 0 (préserve phase de vol running/CMJ).
            # Env var NO_FLOOR_CLAMP=1 pour désactiver (edge case descente
            # d'escalier / pente descendante).
            import os as _os_local
            if not bool(int(_os_local.environ.get("NO_FLOOR_CLAMP", "0"))):
                clamp_shifts = np.zeros(kpts.shape[0], dtype=np.float64)
                n_clamped = 0
                all_min_y = []
                for i in range(kpts.shape[0]):
                    foot = kpts[i, _FOOT_INDICES]
                    if np.any(np.isnan(foot)):
                        continue
                    min_y = float(np.min(foot[:, 1]))
                    all_min_y.append(min_y)
                    if min_y < 0.0:
                        clamp_shifts[i] = -min_y  # shift UP by |min_y|
                        n_clamped += 1
                if all_min_y:
                    print(f"  [floor clamp] foot markers Y range = "
                          f"[{min(all_min_y)*100:+.1f}, {max(all_min_y)*100:+.1f}] cm "
                          f"across {len(all_min_y)} frames "
                          f"({n_clamped} needed clamp)")
                if n_clamped > 0:
                    kpts[:, :, 1] += clamp_shifts[:, None]
                    if jc is not None:
                        jc[:, :, 1] += clamp_shifts[:, None]
                    max_clamp = float(np.max(clamp_shifts))
                    print(f"  [floor clamp] ground-penetration clamp APPLIED to "
                          f"{n_clamped}/{kpts.shape[0]} frames (max shift +{max_clamp*100:.1f} cm)")
                    self._last_penetration_clamp_m = clamp_shifts.copy()

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
                              Points en CAMERA LOCAL frame (no cam_t added).
                              The cam_t contribution is replayed internally via
                              `_last_xz_deltas_m` so that the result matches
                              what transform() produces for the kpts. Adding
                              cam_t to the input here would bake a residual
                              cam_t.Y offset into the mesh frame that does not
                              exist in the kpts/TRC/anatomical chain.
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
                # Stationary mode : verts d'entrée sont RAW (sans cam_t baked).
                # On rejoue la même Y-injection que transform() fait aux kpts
                # pour que mesh et anatomical/kpts restent verticalement
                # alignés (squat descend, etc).
                if (self._last_stationary_cam_t_y_m is not None
                        and i < len(self._last_stationary_cam_t_y_m)):
                    w[:, 1] += self._last_stationary_cam_t_y_m[i]
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
            # Rejoue le penetration clamp per-frame (fix STS pieds au sol)
            if self._last_penetration_clamp_m is not None:
                for i, w in enumerate(pre_ground):
                    if w is None or i >= len(self._last_penetration_clamp_m):
                        continue
                    w[:, 1] += self._last_penetration_clamp_m[i]
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
        # Utilise la NORME 3D ||c_head - foot|| au lieu de la projection Y.
        # Sinon, sur les vidéos avec pitch caméra (e.g. MoGe +18°), le sujet
        # raw est INCLINÉ → c_head_y - foot_y = vraie_taille × cos(pitch) <
        # vraie_taille. Le scale = subject_height / (vraie_taille × cos) est
        # trop grand. Après application du scale et de la rotation correctrice,
        # le sujet apparait sur-scalé par 1/cos(pitch) (e.g. +5% à 18°). La
        # norme 3D est invariante par rotation → scale correct quelle que soit
        # l'inclinaison du sujet/sol dans le brut.
        heights = []
        N = kpts.shape[0]
        for i in range(N):
            foot_idx = np.argmin(kpts[i, _FOOT_INDICES, 1])
            foot = kpts[i, _FOOT_INDICES[foot_idx]]

            if jc is not None:
                top = jc[i, _JCOORDS_HEAD_IDX]
            else:
                # Fallback nose (kpt 0). Project to "head top" via nose height ratio
                # (kept on Y norm because the fallback is rarely used).
                nose_y = kpts[i, 0, 1]
                foot_y = foot[1]
                h = (nose_y - foot_y) / self._NOSE_HEIGHT_FRACTION
                if h > 0.1:
                    heights.append(h)
                continue

            h = float(np.linalg.norm(top - foot))
            if h > 0.1:
                heights.append(h)

        if not heights:
            return 1.0
        # percentile 95 au lieu de mean : sur les vidéos avec flexion (squat,
        # sit-to-stand, etc.) c_head_y descend pendant le mouvement et la
        # moyenne sous-estime la vraie hauteur du sujet debout. Le 95ᵉ
        # percentile approxime la hauteur "debout droit" (Pose2Sim utilise une
        # static window pour le ScaleTool OpenSim, on fait l'équivalent ici
        # côté CoordinateTransformer pour ne pas sur-scaler le sujet).
        return self.subject_height / float(np.percentile(heights, 95))

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

    def _contact_aware_ground(
        self,
        kpts: np.ndarray,
        fps: float = 30.0,
        return_offsets: bool = False,
        band_m: float = 0.05,
        max_flight_s: float = 0.6,
        ground_pct: float = 10.0,
    ):
        """Ancrage sol conscient du contact.

        Problème : la repro monoculaire pose le bassin à hauteur ~fixe (aucun
        mouvement vertical global) ; en squat les pieds "montent" au lieu que le
        bassin descende. `_align_to_ground` (plaque le pied bas à 0 CHAQUE frame)
        corrige le squat mais tue la phase de vol (course/saut → "saute").

        Ici on distingue par la DURÉE de l'excursion des pieds au-dessus du sol :
        - **contact soutenu** (pieds au-dessus du sol longtemps = squat inversé,
          ou pied planté) → on re-ancre par frame (offset = pied le plus bas) →
          le bassin descend correctement.
        - **vol bref** (< ``max_flight_s`` = course/saut) → on TIENT l'offset au
          niveau du sol (interpolé décollage→réception) → les pieds "flottent"
          pendant le vol, pas de yank vers le bas.

        Args:
            kpts : (N, K, 3) en mètres, OpenSim world (Y-up), pas encore ancré.
            fps  : cadence (pour le seuil de durée du vol).
        Returns:
            (kpts_ancré, offsets)  si return_offsets, sinon kpts_ancré.
            offsets[t] = décalage Y soustrait à la frame t.
        """
        result = kpts.copy()
        T = kpts.shape[0]
        min_y = np.array([float(np.min(kpts[i, _FOOT_INDICES, 1])) for i in range(T)],
                         dtype=np.float64)
        finite = np.isfinite(min_y)
        if not finite.any():
            offsets = np.zeros(T)
            return (result, offsets) if return_offsets else result
        # Niveau du sol = bas robuste des pieds sur tout le clip.
        ground = float(np.nanpercentile(min_y[finite], ground_pct))
        offsets = min_y.copy()
        airborne = (min_y > ground + band_m) & finite
        max_flight_frames = max(1, int(round(max_flight_s * fps)))
        i = 0
        while i < T:
            if airborne[i]:
                j = i
                while j < T and airborne[j]:
                    j += 1
                if (j - i) <= max_flight_frames:
                    # Vol bref → tenir le sol : interp niveau décollage→réception.
                    lo = min_y[i - 1] if i > 0 and finite[i - 1] else ground
                    hi = min_y[j] if j < T and finite[j] else ground
                    offsets[i:j] = np.linspace(lo, hi, j - i)
                # else : contact soutenu (squat) → garder offsets=min_y (re-ancrage)
                i = j
            else:
                i += 1
        # Frames non-finies : pas de shift.
        offsets[~finite] = 0.0
        for i in range(T):
            result[i, :, 1] -= offsets[i]
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
    def _frame_sharpness(frame_bgr) -> float:
        """Variance du Laplacien = mesure de netteté monoculaire.

        Empiriquement : > 100 net, 60-100 acceptable, < 60 flou (motion blur,
        autofocus non convergé, compression H.264 keyframe médiocre).
        """
        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _ransac_plane_normal(
        pts: np.ndarray,
        n_iter: int = 100,
        dist_thresh: float = 0.05,
        min_inliers: int = 50,
    ) -> tuple:
        """RANSAC plane fit → (normal_refined_or_None, n_inliers).

        Fait N itérations : sample 3 points → plan → count inliers within
        `dist_thresh` meters. Le meilleur set d'inliers est ensuite raffiné
        par SVD pour un normal stable.

        Résistant aux outliers (jusqu'à ~40-50% de points hors sol).
        """
        N = len(pts)
        if N < 3:
            return None, 0
        rng = np.random.default_rng(seed=42)
        best_normal = None
        best_inliers = 0
        best_pt = None
        for _ in range(n_iter):
            idx = rng.choice(N, 3, replace=False)
            p1, p2, p3 = pts[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal = normal / norm
            if normal[1] < 0:
                normal = -normal
            d = np.abs((pts - p1) @ normal)
            inliers = int((d < dist_thresh).sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_pt = p1
        if best_normal is None or best_inliers < min_inliers:
            return None, best_inliers
        # Refine with SVD on inliers set for stable normal
        d = np.abs((pts - best_pt) @ best_normal)
        inlier_pts = pts[d < dist_thresh]
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        refined = Vt[-1]
        if refined[1] < 0:
            refined = -refined
        return refined, best_inliers

    @staticmethod
    def floor_angle_from_moge_points(
        points: np.ndarray,
        mask: np.ndarray,
        person_bbox=None,
        orig_hw=None,
        floor_frac: float = 0.25,
        n_samples: int = 4000,
        return_raw: bool = False,
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

        # STRATÉGIE "sous-bbox" (idée Florian 2026-07-08) : si bbox dispo, on
        # restreint les candidats sol à la bande de pixels SOUS le sujet, pas
        # juste "bottom 25% de l'image". Physiquement : les pieds touchent le
        # sol donc les pixels immédiatement sous la bbox = sol garanti.
        # Fallback (pas de bbox) : ancien comportement "bottom floor_frac".
        floor_row_min = None   # ligne min (exclusive) du floor band ; None = pas de contrainte pixel
        floor_row_fallback = None  # fallback "à côté" (mi-bbox) si pas assez de sol dessous
        if person_bbox is not None and orig_hw is not None:
            oh, ow = orig_hw
            x1, y1, x2, y2 = person_bbox
            # Scale bbox from original frame to MoGe grid
            gx1 = int(x1 / ow * W)
            gy1 = int(y1 / oh * H)
            gx2 = int(x2 / ow * W)
            gy2 = int(y2 / oh * H)
            # Exclude bbox area (subject) + 10% margin
            margin_x = max(1, int((gx2 - gx1) * 0.10))
            margin_y = max(1, int((gy2 - gy1) * 0.10))
            gx1_e = max(0, gx1 - margin_x)
            gy1_e = max(0, gy1 - margin_y)
            gx2_e = min(W - 1, gx2 + margin_x)
            gy2_e = min(H - 1, gy2 + margin_y)
            valid_mask[gy1_e:gy2_e+1, gx1_e:gx2_e+1] = False
            # Band candidats sol = lignes SOUS la bbox (pieds → sol garanti).
            # On garde une marge de 5% sous la bbox pour éviter les artefacts
            # de chaussures / ombre. Env var MOGE_BBOX_FLOOR=0 pour désactiver.
            use_bbox_floor = bool(int(_os.environ.get("MOGE_BBOX_FLOOR", "1")))
            if use_bbox_floor:
                below_margin = max(1, int((gy2 - gy1) * 0.05))
                floor_row_min = min(H - 1, gy2 + below_margin)
                # Fallback : depuis la moitié basse de la bbox → capte le sol
                # À CÔTÉ des jambes/pieds (bbox intérieur déjà exclu du mask).
                floor_row_fallback = max(0, gy1 + int(0.5 * (gy2 - gy1)))

        # Restreint le mask aux lignes SOUS la bbox si applicable ; si trop peu
        # de pixels sol dessous (sujet en bas du cadre) → fallback "à côté".
        if floor_row_min is not None:
            band = np.zeros_like(valid_mask)
            band[floor_row_min:, :] = True
            vm_below = valid_mask & band
            if vm_below.sum() < 200 and floor_row_fallback is not None:
                band_side = np.zeros_like(valid_mask)
                band_side[floor_row_fallback:, :] = True
                valid_mask = valid_mask & band_side
            else:
                valid_mask = vm_below

        # Flatten to valid points
        pts_flat = points.reshape(-1, 3).astype(np.float64)
        mask_flat = valid_mask.reshape(-1)
        valid = pts_flat[mask_flat]
        if len(valid) < 50:
            return (0.0, 0.0)

        if floor_row_min is not None:
            # En mode "sous-bbox" : tous les points restants sont candidats sol.
            floor_pts = valid
        else:
            # Fallback : sélection depth-independent par ligne image.
            # BUG DÉCOUVERT 2026-07-08 : le comment initial disait "Y-UP" mais
            # MoGe utilise en fait Y-DOWN (OpenCV convention). Avec Y-UP on
            # sélectionnait le haut de l'image (mur/ciel), pas le sol.
            # Y-DOWN par défaut. Env var MOGE_Y_DOWN=0 pour revenir à l'ancien.
            y_norm = valid[:, 1] / valid[:, 2]  # Y_cam / Z_cam
            if bool(int(_os.environ.get("MOGE_Y_DOWN", "1"))):
                # Y down (OpenCV) : sol = Y_cam > 0 → top of y_norm
                thresh = np.percentile(y_norm, (1.0 - floor_frac) * 100)
                floor_pts = valid[y_norm >= thresh]
            else:
                thresh = np.percentile(y_norm, floor_frac * 100)
                floor_pts = valid[y_norm <= thresh]

        if len(floor_pts) < 20:
            return (0.0, 0.0)

        # Random subsample for speed
        if len(floor_pts) > n_samples:
            idx = np.random.choice(len(floor_pts), n_samples, replace=False)
            floor_pts = floor_pts[idx]

        # Plane fit : RANSAC par défaut (robuste aux outliers), SVD en fallback.
        # Env var MOGE_PLANE_FIT=svd pour forcer l'ancien comportement si régression.
        import os as _os
        fit_method = _os.environ.get("MOGE_PLANE_FIT", "ransac").lower()
        if fit_method == "ransac":
            ransac_normal, n_inl = CoordinateTransformer._ransac_plane_normal(
                floor_pts, n_iter=200, dist_thresh=0.05, min_inliers=50)
            if ransac_normal is not None:
                normal = ransac_normal
            else:
                # RANSAC failed → SVD fallback
                centroid = floor_pts.mean(axis=0)
                _, _, Vt = np.linalg.svd(floor_pts - centroid, full_matrices=False)
                normal = Vt[-1]
                if normal[1] < 0:
                    normal = -normal
        else:
            centroid = floor_pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(floor_pts - centroid, full_matrices=False)
            normal = Vt[-1]
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
        clipped_pitch = float(np.clip(correction_pitch, -60.0, 60.0))
        clipped_roll = 0.0 if DISABLE_ROLL else float(np.clip(-correction_roll, -60.0, 60.0))
        if return_raw:
            return (clipped_pitch, clipped_roll, float(raw_pitch), float(raw_roll))
        return (clipped_pitch, clipped_roll)

    @staticmethod
    def robust_floor_angle_multi_frame(
        video_path: str,
        depth_estimator_fn,
        person_bbox_fn=None,
        n_samples: int = 8,
        sharpness_min: float = 30.0,
        max_std_deg: float = 5.0,
        max_raw_pitch_deg: float = 45.0,
        max_raw_roll_deg: float = 45.0,
    ) -> tuple:
        """Estime le plan sol de manière robuste sur multi-frames.

        Sample N frames dispersées, filtre par netteté (variance Laplacien),
        fit RANSAC → median pitch/roll + std check.

        Args:
            video_path : chemin vidéo à échantillonner
            depth_estimator_fn : callable(frame_rgb) -> (pts, mask) pour MoGe
            person_bbox_fn : callable(frame_rgb) -> bbox or None (exclut sujet).
                Optionnel — si None, aucune exclusion (léger biais si sujet visible
                en bas d'image).
            n_samples : nb de frames à échantillonner (dispersées uniformément)
            sharpness_min : seuil Laplacien variance (< = frame skippée)
            max_std_deg : si std des N pitchs > ce seuil, considère "unstable"

        Returns:
            (pitch_deg, roll_deg, quality_dict) où quality_dict contient :
              - status : "ok" | "unstable" | "insufficient_samples"
              - n_used : nb de frames retenues
              - pitch_std_deg, roll_std_deg
              - per_frame : liste des estimations (debug)
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0, 0.0, {"status": "cannot_open_video"}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total < 2:
            cap.release()
            return 0.0, 0.0, {"status": "video_too_short"}

        # Sample n_samples frames uniformly, skipping first/last 5% (motion
        # blur + freeze frames)
        margin = max(1, int(total * 0.05))
        sample_idx = np.linspace(margin, total - margin - 1, n_samples).astype(int)
        sample_idx = np.unique(sample_idx)

        per_frame = []
        pitches, rolls, weights = [], [], []
        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            sharp = CoordinateTransformer._frame_sharpness(frame_bgr)
            if sharp < sharpness_min:
                per_frame.append({"frame": int(idx), "sharpness": sharp,
                                  "status": "skipped_blurry"})
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                pts, mask = depth_estimator_fn(frame_rgb)
            except Exception as e:
                per_frame.append({"frame": int(idx), "sharpness": sharp,
                                  "status": f"depth_fail:{str(e)[:60]}"})
                continue

            person_bbox = None
            if person_bbox_fn is not None:
                try:
                    person_bbox = person_bbox_fn(frame_rgb)
                except Exception:
                    person_bbox = None

            orig_hw = (frame_bgr.shape[0], frame_bgr.shape[1])
            try:
                p, r, raw_p, raw_r = CoordinateTransformer.floor_angle_from_moge_points(
                    pts, mask, person_bbox=person_bbox, orig_hw=orig_hw,
                    return_raw=True)
            except Exception as e:
                per_frame.append({"frame": int(idx), "sharpness": sharp,
                                  "status": f"floor_fail:{str(e)[:60]}"})
                continue

            # Sanity check : un smartphone tenu à hauteur d'homme donne un raw
            # pitch < 30-40°. > 45° = MoGe a détecté un mur, un panneau, ou une
            # surface non-sol. Frame rejetée.
            if abs(raw_p) > max_raw_pitch_deg or abs(raw_r) > max_raw_roll_deg:
                per_frame.append({
                    "frame": int(idx), "sharpness": sharp,
                    "raw_pitch": raw_p, "raw_roll": raw_r,
                    "status": "rejected_aberrant_plane",
                })
                continue

            per_frame.append({
                "frame": int(idx), "sharpness": sharp,
                "pitch": p, "roll": r,
                "raw_pitch": raw_p, "raw_roll": raw_r,
                "status": "ok",
            })
            pitches.append(p)
            rolls.append(r)
            weights.append(sharp)

        cap.release()

        if len(pitches) < 3:
            return 0.0, 0.0, {
                "status": "insufficient_samples",
                "n_used": len(pitches),
                "n_requested": len(sample_idx),
                "per_frame": per_frame,
            }

        pitches_arr = np.array(pitches)
        rolls_arr = np.array(rolls)
        pitch_med = float(np.median(pitches_arr))
        roll_med = float(np.median(rolls_arr))
        pitch_std = float(np.std(pitches_arr))
        roll_std = float(np.std(rolls_arr))

        status = "ok" if pitch_std <= max_std_deg else "unstable"

        return pitch_med, roll_med, {
            "status": status,
            "n_used": len(pitches),
            "n_requested": len(sample_idx),
            "pitch_median": pitch_med,
            "roll_median": roll_med,
            "pitch_std_deg": pitch_std,
            "roll_std_deg": roll_std,
            "per_frame": per_frame,
        }

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

    def _apply_global_translation(self, keypoints, camera_translation, scale,
                                    lock_lateral: bool = False):
        """Apply per-frame XZ translation from cam_t.

        Args:
            lock_lateral : si True, détecte l'axe d'avance principal via PCA
                sur la trajectoire XZ du sujet, ne garde que la composante
                longitudinale (= avance), zéro la composante latérale. Idéal
                pour running/marche/sprint filmés en ligne droite où le bruit
                monoculaire fait "zig-zag" le sujet perpendiculairement.
        """
        num_frames = keypoints.shape[0]
        if camera_translation.ndim == 1:
            camera_translation = np.tile(camera_translation, (num_frames, 1))
        cam_t_opensim = camera_translation @ self.CAMERA_TO_OPENSIM.T * scale
        cam_t_smoothed = self._smooth_cam_t(cam_t_opensim)
        first_frame_t = cam_t_smoothed[0].copy()

        deltas_xz = cam_t_smoothed[:, [0, 2]] - first_frame_t[[0, 2]]  # (N, 2)

        if lock_lateral and num_frames >= 3:
            # PCA 2D sur la trajectoire XZ pour détecter l'axe d'avance
            centered = deltas_xz - deltas_xz.mean(axis=0, keepdims=True)
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            fwd_axis = eigvecs[:, -1]  # eigenvector du plus grand eigenvalue
            fwd_var = float(eigvals[-1])
            lat_var = float(eigvals[0])
            # Aligner le signe pour que fwd_axis matche le sens de progression
            # (fin - début doit avoir une projection positive sur fwd_axis)
            traj_vec = deltas_xz[-1] - deltas_xz[0]
            if np.dot(traj_vec, fwd_axis) < 0:
                fwd_axis = -fwd_axis
            # Ratio variance longitudinale / latérale
            ratio = fwd_var / (lat_var + 1e-9)
            print(f"  [lock_lateral] fwd_axis=({fwd_axis[0]:+.3f}, {fwd_axis[1]:+.3f}) "
                  f"in (X, Z), fwd_var/lat_var={ratio:.1f} (>10 = ligne droite claire)")
            # Projet chaque delta sur fwd_axis, ne garde que cette composante
            proj = deltas_xz @ fwd_axis  # (N,)
            deltas_xz = proj[:, None] * fwd_axis[None, :]  # (N, 2)

        xz_deltas = np.zeros((num_frames, 3))
        for i in range(num_frames):
            keypoints[i, :, 0] += deltas_xz[i, 0]
            keypoints[i, :, 2] += deltas_xz[i, 1]
            xz_deltas[i, 0] = deltas_xz[i, 0]
            xz_deltas[i, 2] = deltas_xz[i, 1]
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


def anti_foot_skate_markers(
    markers_array: np.ndarray,
    marker_names: list,
    fps: float = 30.0,
    band_m: float = 0.06,
    ground_pct: float = 10.0,
    min_contact_s: float = 0.08,
    max_gap_s: float = 0.12,
    max_contact_speed_ms: float = 0.30,
    exit_band_ratio: float = 1.6,
    vel_gate_window_s: float = 0.30,
    vel_gate_std_m: float = 0.02,
    recede_tol_ms: float = 0.05,
    walk_min_travel_m: float = 0.40,
    stance_band_fraction: float = 0.35,
):
    """Anti-glisse pied sur les marqueurs FINAUX du TRC (markerset Flodelaplace).

    À appeler APRÈS le convertisseur (markers_array = (T, M, 3) en mètres,
    Y-up) et AVANT l'export TRC / l'IK : le TRC anti-slidé nourrit ensuite IK +
    retargeting avatar, donc le squelette post-IK et l'avatar sont déjà anti-
    slidés (pas de lavage par l'IK).

    Contact détecté PAR LA HAUTEUR (pied bas soutenu ≥ min_contact_s), pas par la
    vitesse (la dérive à corriger est de la gigue rapide). Pendant chaque phase de
    contact d'un pied, on fige la position XZ de ses marqueurs à la médiane de la
    phase → pied planté.

    Returns:
        (markers_corrigés, shifts) où shifts[t] = (dx, dz) moyen appliqué à la
        frame t (utile pour translater le mesh GLB de la même façon).
    """
    if markers_array is None or markers_array.ndim != 3 or markers_array.shape[0] < 2:
        return markers_array, None
    result = markers_array.copy()
    T = result.shape[0]
    # Auto-détection unité (mm vs m) : les marqueurs Flodelaplace peuvent être
    # en mm (TRC) ou en mètres. band_m est en mètres → on l'échelle.
    _unit = 1000.0 if np.nanmax(np.abs(markers_array)) > 50.0 else 1.0
    band = band_m * _unit
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    # Groupes de marqueurs pied par côté (repères sol : talon + orteil).
    # Repères de pied : on prend toute la semelle disponible plutôt que le seul
    # couple talon/orteil. Un appui se juge mieux sur 5 points (calcanéus,
    # orteil, 5e méta, malléoles) que sur 2, et les marqueurs de semelle SOLE
    # (`*_s1`..`*_s7`, absents des TRC anciens) sont pris s'ils existent.
    # Jeu repris de l'étage ground_anchor de Mesh2Sim, rodé sur de la marche.
    _base = ("CAL", "TOE", "MT5", "LMAL", "MMAL")
    side_markers = {}
    for side in ("L", "R"):
        idx = [name_to_idx[side + n] for n in _base if side + n in name_to_idx]
        idx += [i for n, i in name_to_idx.items()
                if n.startswith(side) and len(n) > 2 and n[1] == "s" and n[2:].isdigit()]
        side_markers[side] = idx
    min_len = max(2, int(round(min_contact_s * fps)))

    # Axe de marche et position du bassin — nécessaires au critère de recul.
    # L'axe est la direction principale du déplacement du bassin (ACP sur XZ),
    # orientée dans le sens du trajet net. S'il n'y a pas de déplacement franc
    # (exercice en place), l'axe n'a pas de sens : on le laisse à None et la
    # détection retombe sur hauteur + stabilité verticale, qui suffisent là.
    pelvis_xz = None
    walk_axis = None
    _pel = [name_to_idx[m] for m in ("RASI", "LASI", "RPSI", "LPSI")
            if m in name_to_idx]
    if _pel:
        pelvis_xz = np.nanmean(result[:, _pel, :][:, :, [0, 2]], axis=1)
        d = pelvis_xz - np.nanmean(pelvis_xz, axis=0)
        d = d[np.all(np.isfinite(d), axis=1)]
        if len(d) > 4:
            net = np.linalg.norm(np.nanmean(pelvis_xz[-5:], axis=0)
                                 - np.nanmean(pelvis_xz[:5], axis=0))
            if net > walk_min_travel_m * _unit:
                _, _, vt = np.linalg.svd(d, full_matrices=False)
                axis = vt[0] / (np.linalg.norm(vt[0]) or 1.0)
                # oriente l'axe dans le sens de la marche
                if (np.nanmean(pelvis_xz[-5:], axis=0)
                        - np.nanmean(pelvis_xz[:5], axis=0)) @ axis < 0:
                    axis = -axis
                walk_axis = axis

    # Contact + position XZ par côté.
    foot_xz = {}
    foot_y = {}
    contact = {}
    for side, idx in side_markers.items():
        if not idx:
            continue
        # Point de contact = marqueur du pied le PLUS BAS par frame (talon OU
        # orteil selon la pose : extension pointe de pied → seul l'orteil touche).
        sub = result[:, idx, :]              # (T, k, 3)
        foot = np.full((T, 3), np.nan, dtype=np.float64)
        for t in range(T):
            col = sub[t, :, 1]
            if np.all(np.isnan(col)):
                continue
            foot[t] = sub[t, int(np.nanargmin(col))]
        y = foot[:, 1]
        finite = np.all(np.isfinite(foot), axis=1)
        if not finite.any():
            continue
        ground = float(np.nanpercentile(y[finite], ground_pct))
        # Bande ADAPTATIVE (Mesh2Sim) : une bande fixe de 6 cm est absurde quand
        # la reconstruction ne donne que 5 cm de garde au sol — le pied est alors
        # "en contact" en permanence. On la prend proportionnelle à l'amplitude
        # REELLE du pied sur le clip, avec band_m comme plancher.
        _span = float(np.nanpercentile(y[finite], 85.0)) - ground
        band = max(band_m * _unit, stance_band_fraction * _span)

        # Hystérésis : on ENTRE en contact plus bas qu'on n'en SORT. Avec un
        # seuil unique, un pied posé qui oscille d'un millimètre autour de la
        # limite bascule sans arrêt entre appui et vol.
        lo = y < ground + band
        hi = y < ground + band * exit_band_ratio
        c = np.zeros(T, dtype=bool)
        state = False
        for t in range(T):
            state = (hi[t] if state else lo[t])
            c[t] = state
        c &= finite

        # Porte de vitesse : un pied réellement en appui est STABLE en hauteur.
        # On exige que l'écart-type glissant de sa distance au sol reste faible.
        # C'est ce qui distingue « le pied a vraiment décollé » de « l'inférence
        # a bougé » — sans ça, une secousse de tracking passe pour un envol et
        # la correction accumulée se relâche d'un coup.
        win = max(3, int(round(vel_gate_window_s * fps)))
        pad = win // 2
        ypad = np.pad(np.where(finite, y, np.nan), pad, mode="edge")
        roll = np.lib.stride_tricks.sliding_window_view(ypad, win)[:T]
        with np.errstate(invalid="ignore"):
            stable = np.nan_to_num(np.nanstd(roll, axis=1), nan=np.inf) < (
                vel_gate_std_m * _unit)
        c &= stable

        # Critère de RECUL RELATIF (Mesh2Sim, _contacts_on_axis).
        #
        # La vitesse absolue du pied est inutilisable : la trajectoire globale
        # vient de l'estimation caméra et son bruit domine tout (mesuré sur un
        # couloir de marche : bassin à 6,4 m/s de pointe pour une marche à
        # 1,9 m/s). Un pied planté n'y paraît jamais immobile.
        #
        # Le signe du mouvement RELATIF au bassin, lui, est robuste : le bruit
        # commun s'annule par différence. Un pied en appui RECULE par rapport au
        # bassin pendant que le corps avance ; un pied en vol AVANCE. C'est ce
        # qui discrimine appui et oscillation, pas la hauteur.
        if walk_axis is not None and pelvis_xz is not None:
            rel = (np.nanmean(sub[:, :, [0, 2]], axis=1) - pelvis_xz) @ walk_axis
            fwd = np.zeros(T)
            fwd[1:] = np.diff(rel) * fps
            c &= fwd <= (recede_tol_ms * _unit)

        # Bouche les DÉCROCHAGES trop courts pour être un vol réel. Le filtre
        # min_contact_s ci-dessous ne rejette que les contacts trop courts ; il
        # n'avait pas de symétrique. Résultat : une gigue de 1-2 frames au-dessus
        # du seuil cassait une phase d'appui en deux, et au raccord le décalage
        # accumulé se relâchait d'un coup — le pied « décrochait » et le corps
        # sautait brusquement alors que le sujet n'avait jamais quitté le sol.
        # On ferme donc les trous < max_gap_s AVANT le filtre de durée.
        # Les trous en BORD de séquence sont laissés tels quels : un vrai vol au
        # début ou à la fin du clip ne doit pas être comblé par extrapolation.
        max_gap = max(1, int(round(max_gap_s * fps)))
        i = 0
        while i < T:
            if not c[i]:
                j = i
                while j < T and not c[j]:
                    j += 1
                if i > 0 and j < T and (j - i) <= max_gap:
                    c[i:j] = True
                i = j
            else:
                i += 1

        # ne garde que les runs de contact ≥ min_len
        keep = np.zeros(T, dtype=bool)
        i = 0
        while i < T:
            if c[i]:
                j = i
                while j < T and c[j]:
                    j += 1
                if (j - i) >= min_len:
                    keep[i:j] = True
                i = j
            else:
                i += 1
        # ⚠️ La HAUTEUR vient du marqueur le plus bas (talon OU orteil selon la
        # pose) — c'est correct pour détecter le contact. Mais le DÉPLACEMENT
        # doit se mesurer sur une référence STABLE : si l'on suit « le plus bas »,
        # une bascule talon→orteil fait sauter la référence d'un marqueur à
        # l'autre, et la distance talon-orteil (~22 cm) est appliquée telle quelle
        # comme décalage du corps. Le sujet part brutalement de côté alors qu'il
        # n'a jamais quitté le sol. On prend donc le centroïde XZ des marqueurs
        # du pied, insensible à la bascule.
        foot_xz[side] = np.nanmean(sub[:, :, [0, 2]], axis=1)
        foot_y[side] = foot[:, 1]
        contact[side] = keep

    if not contact:
        return result, np.zeros((T, 2))

    # ANCRAGE PAR PHASE D'APPUI (portage de l'étage ground_anchor de Mesh2Sim,
    # rodé sur de la marche).
    #
    # La version précédente corrigeait le déplacement du pied ENTRE DEUX IMAGES.
    # Elle lissait donc la dérive sans jamais PLANTER le pied : l'erreur
    # résiduelle s'accumulait librement le long d'une phase d'appui, et sur de
    # la marche le pied glissait de plusieurs dizaines de centimètres.
    #
    # Ici, dès qu'un pied se pose on mémorise une ancre, et le décalage du corps
    # est celui qui ramène ce pied SUR son ancre — une consigne absolue, pas une
    # dérivée. L'ancre est posée en coordonnées DÉJÀ CORRIGÉES
    # (`foot + shift` à l'instant de la pose) : sans cela, chaque nouveau pas
    # exigerait d'annuler tout le décalage accumulé et rappellerait le corps en
    # arrière. Le pied qui décolle libère son ancre, donc la marche progresse.
    shifts = np.zeros((T, 2), dtype=np.float64)
    # Plafond exprimé en VITESSE (donc indépendant du fps). Volontairement
    # serré : notre déplacement global vient de `cam_t`, et une correction
    # ample reviendrait à reconstruire la trajectoire depuis les pieds —
    # ce que fait Mesh2Sim, mais qui entrerait ici en conflit avec cam_t
    # (mesuré : 3,95 m de correction sur un couloir de 10,8 m). On se limite
    # donc à retirer la gigue.
    max_step = max_contact_speed_ms * _unit / max(fps, 1e-6)
    anchors: dict[str, np.ndarray | None] = {s: None for s in contact}

    for t in range(T):
        needed = []
        for side in contact:
            xz = foot_xz[side][t]
            if not (contact[side][t] and np.all(np.isfinite(xz))):
                anchors[side] = None      # pied en vol → l'ancre expire
                continue
            if anchors[side] is None:     # pose : ancre = position corrigée
                anchors[side] = xz + shifts[t - 1] if t > 0 else xz.copy()
            needed.append(anchors[side] - xz)

        if not needed:
            shifts[t] = shifts[t - 1] if t > 0 else 0.0
            continue

        # Double appui : moyenne des deux consignes. Suivre un seul pied ferait
        # basculer le corps à chaque transition d'appui.
        target = np.mean(np.stack(needed), axis=0)
        step = target - (shifts[t - 1] if t > 0 else 0.0)
        # Un pied en appui ne glisse pas vite : au-delà, c'est de la gigue
        # d'inférence, et la compenser d'un coup projetterait le corps de côté.
        n = float(np.linalg.norm(step))
        if n > max_step:
            step *= max_step / n
        shifts[t] = (shifts[t - 1] if t > 0 else 0.0) + step

    # Lissage zéro-phase de la correction cumulée : les transitions d'appui
    # laissent des angles vifs dans la trajectoire du bassin. 2 Hz laisse passer
    # la cadence de marche (~1 Hz) tout en supprimant les ruptures.
    if T > 18 and fps > 6:
        try:
            from scipy.signal import butter, sosfiltfilt
            wn = min(2.0 / (fps / 2.0), 0.99)
            shifts = sosfiltfilt(butter(2, wn, btype="low", output="sos"),
                                 shifts, axis=0)
        except Exception:
            pass  # scipy absent ou signal trop court : on garde le brut

    result[:, :, 0] += shifts[:, 0][:, None]
    result[:, :, 2] += shifts[:, 1][:, None]
    # shifts retournés en MÈTRES (pour appliquer le même décalage au mesh GLB,
    # qui est en mètres), quelle que soit l'unité de markers_array.
    return result, shifts / _unit