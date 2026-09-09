"""
Coordinate system transformation from camera to OpenSim.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import uniform_filter1d

# jcoords index for the head joint (c_head) in the MHR 127-joint armature
_JCOORDS_HEAD_IDX = 113

# Chaine articulaire cheville -> sommet du crane, dans les 127 joints du rig MHR.
# Sert a estimer la STATURE du sujet par somme de segments : contrairement a une
# distance tete-pied a vol d'oiseau, une somme de segments est INVARIANTE PAR
# POSTURE. Le rachis est somme vertebre par vertebre, sinon un tronc penche
# raccourcit la mesure et on retombe sur le meme biais.
_STATURE_CHAIN: tuple[tuple[int, int], ...] = (
    (20, 19),    # tibia (cheville -> genou)
    (19, 18),    # femur (genou -> hanche)
    (18, 1),     # bassin
    (1, 34), (34, 35), (35, 36), (36, 37), (37, 110),  # rachis
    (110, 113),  # cou -> tete
    (113, 126),  # tete -> sommet du crane
)
# Topologie verifiee sur mhr_skeleton.json (joint_parents) : chaque maillon est
# bien une relation parent-enfant directe. Deux pieges evites :
#   - 18 et 34 sont des branches SOEURS (parent commun = 1), il n'existe aucun
#     lien 18 -> 35 ; on passe donc par le joint 1.
#   - 112 est un FRERE de 113, pas un maillon du rachis. L'y inclure etait
#     numeriquement indolore (3,7 mm) mais topologiquement faux.
# stature = somme_chaine x C. Mesure sur le template MHR (betas = 0) : chaine
# 1,7218 m pour une stature de 1,7657 m. C absorbe la courbure du rachis et la
# hauteur cheville-sol, toutes deux proportionnelles a l'echelle du sujet.
_STATURE_CHAIN_TO_HEIGHT = 1.0255

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

    # Normales de sol par image, en repere camera OpenCV, remplies par
    # `robust_floor_angle_multi_frame` quand MOGE_WORLD_FRAME=1. Attribut de
    # CLASSE parce que l estimation multi-images est statique : elle tourne
    # avant toute instanciation du transformateur, et c est le seul endroit ou
    # la grille de points MoGe existe encore.
    _ups_cam_m2s: list = []          # [(idx_image, up_camera)]
    _gc_up_world_m2s = None          # up monde agrege (retro-compat)
    _gc_ups_world_m2s: list = []     # [(idx_image, up_monde)] par image-cle
    _n_frames_video: int = 0

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
        stable_floor: bool = False,
        plantar_indices: Optional[np.ndarray] = None,
        stable_floor_drift: str = "linear+stance",
        ground_margin_m: float = 0.01,
        fps: float = 30.0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform keypoints (and optionally jcoords) to OpenSim world space.

        Args:
            keypoints_3d : (N, 70, 3)  MHR70 keypoints in camera space
            jcoords_3d   : (N, 127, 3) MHR armature joints in camera space (optional)

        Args (suite) :
            stable_floor : mise au sol « stable » portee de Mesh2Sim
                (`sam_3d_body/export/ground_plane.py`). Remplace le trio
                clamp per-frame / align_to_ground / contact_anchor par une
                transformation a peu de parametres calculee UNE FOIS pour
                l'essai : rotation globale du plan du sol, derive verticale,
                offset constant. Voir `docs/portage_sol_contact_mesh2sim.md`.
                DEFAUT OFF : le comportement historique est inchange tant que
                l'appelant ne demande rien.
            plantar_indices : indices, DANS LE TABLEAU ``jcoords_3d``, des points
                plantaires (marqueurs SOLE_*). Le pipeline concatene les
                marqueurs anatomiques derriere les 127 joints MHR
                (`demo_video_opensim.py`), donc ces indices valent
                127 + rang du marqueur. Sans eux, ``stable_floor`` retombe sur
                les 4 keypoints CUTANES de ``_FOOT_INDICES``, ce qui pose le sol
                3 cm trop haut et fait battre la reference au rythme du deroule
                du pied — c'est justement ce qu'on cherche a eviter.
            stable_floor_drift : "none", "linear", "stance" ou "linear+stance".

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
            # VERTICALE DE LA TRANSLATION CAMERA — manquait depuis l'origine.
            # `pred_keypoints_3d` est RELATIF A LA RACINE (sam3d_body.py :
            # `j3d_cam = j3d + pred_cam_t` ne sert qu'a la projection 2D), et
            # `_apply_global_translation` n'applique que X et Z. Le bassin
            # restait donc a hauteur constante : sur un squat ce sont les PIEDS
            # qui montaient de 20 cm au fond (`outputs/SQUAT_VIT`, 2026-09-09,
            # « ses jambes remontent pendant le squat »). Le sprint, la marche et
            # la course n'en souffraient pas visiblement : leur bassin ne varie
            # que de 5 a 8 cm. Squat, saut, lever de chaise et hop vivaient en
            # mode stationnaire, ou cette injection existait deja (ci-dessous) ;
            # c'est en sortant le squat du stationnaire qu'elle a manque.
            # Meme injection, meme rejeu cote mesh (`apply_pipeline_to_verts`).
            if not lock_vertical:
                _ct = camera_translation
                if _ct.ndim == 1:
                    _ct = np.tile(_ct, (kpts.shape[0], 1))
                ct_os = (_ct @ self.CAMERA_TO_OPENSIM.T * scale).astype(np.float64)
                kpts[:, :, 1] += ct_os[:, 1:2]
                if jc is not None:
                    jc[:, :, 1] += ct_os[:, 1:2]
                self._last_stationary_cam_t_y_m = ct_os[:, 1].copy()
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
            # MODE REPERE MONDE — DEFAUT depuis le 2026-09-08.
            # `MOGE_WORLD_FRAME=0` revient a l ancienne chaine.
            #
            # Porte la recette Mesh2Sim : normale du sol par image depuis la
            # grille MoGe, orientation GEOMETRIQUE de cette normale, agregation
            # robuste avec rejet au-dela de 8 deg, conversion OpenCV -> monde par
            # (x, -y, -z) — une ROTATION, pas une negation — angle applique EN
            # ENTIER, et rotation GLOBALE unique de la scene plutot qu une
            # rotation par image autour du bassin.
            #
            # ⚠️ FILET DE SECURITE, et pourquoi il est indispensable.
            # Chez Mesh2Sim, l agregation MoGe n est JAMAIS seule : elle doit
            # s accorder a moins de 8 deg avec GeoCalib, sinon la valeur est
            # rejetee. Nous n avons pas ce second estimateur. Or leur agregation
            # prend la MOYENNE des normales avant de rejeter au-dela de 8 deg :
            # quand les images sont tres dispersees, la moyenne est deja tiree
            # par les aberrantes et il ne survit presque rien.
            # Mesure du 2026-09-08 sur `Sprint_start` : 1 image sur 8 gardee,
            # scene inclinee de 32,20 deg, sol stable derriere a 17,29 deg et
            # -97,7 cm d offset. Inexploitable.
            # L ancienne chaine, elle, prend la MEDIANE des angles avec controle
            # d ecart-type — plus robuste a forte dispersion, et c est pourquoi
            # elle tient en production.
            # On garde donc leur agregation telle quelle, et l on retombe sur la
            # mediane des que le consensus est trop maigre pour etre credible.
            import os as _os_wf
            # Remise a zero : ces etats sont rejoues sur le MESH plus tard. Sans
            # cela, une video sans repere monde rejouerait la rotation de la
            # precedente sur ses sommets — meme famille de fuite que
            # `_ups_cam_m2s` dans un worker persistant.
            self._last_world_frame_R = None
            self._last_world_frame_series = None
            self._last_world_frame_dy_m = 0.0
            _deja_redresse = False
            # ⚠️ REMIS EN OPT-IN le 2026-09-09 au soir. Mesure sur `Squat.MP4`,
            # tronc en phase debout, 0 degre = vertical :
            #     repere monde COUPE      10,60 deg
            #     repere monde a 11,94    19,01 deg
            # La rotation ne CORRIGE pas l'inclinaison, elle s'y AJOUTE
            # (10,60 + 11,94 = 22,5, mesure 19,0). Le sujet ressort plus penche
            # qu'avant, ce que Florian a vu a l'oeil avant que je le mesure.
            #
            # Piste, non verifiee : le vecteur estime vaut [-0,031, +0,978,
            # +0,205]. Si le vrai up avait un Z NEGATIF, notre rotation
            # doublerait l'erreur au lieu de l'annuler — famille de piege que
            # Mesh2Sim documente sur les conventions d'axes. A instruire sur une
            # scene dont on connait l'inclinaison, pas au jugé.
            #
            # Tout ce qui a ete mesure ce soir sur le consensus reste vrai et
            # utile : MoGe a pleine resolution s'accorde avec GeoCalib a 0,2 deg
            # sur un squat et 2,9 sur un sprint. C'est l'APPLICATION de cette
            # verticale qui est fausse, pas son estimation.
            _world_frame = (_src == "MoGe"
                            and bool(int(_os_wf.environ.get("MOGE_WORLD_FRAME", "0"))))
            if _world_frame:
                # Les angles recus sont deja multiplies par LEAN_SCALE ; ce mode
                # veut l angle MESURE, donc on defait ce facteur. Si le clamp a
                # 60 deg a mordu, la reconstruction serait fausse — on ne peut
                # pas le savoir ici, mais un pitch a la borne est deja anormal.
                # ── CAMERA MOBILE : verticale interpolee entre images-cles ──
                # Une rotation GLOBALE suppose une camera fixe. Des qu'elle
                # bouge, la verticale n'est pas la meme au debut et a la fin, et
                # le mode global ne peut que moyenner (faux) ou refuser (rien
                # redresse). On bascule donc sur une verticale PAR IMAGE,
                # interpolee entre les images-cles.
                #
                # Le critere de bascule est la DISPERSION elle-meme : sur une
                # camera fixe elle vaut 0,4 a 0,7 deg (sprint, squat), sur un
                # essai in-situ filme a la main 6,9 deg (Titia, 2026-09-09).
                # Faible dispersion, on garde le mode global, eprouve ; forte
                # dispersion, ce n'est pas du bruit mais le mouvement reel de la
                # camera, et l'interpoler le suit au lieu de l'ecraser.
                #
                # Mesure du 2026-09-09 sur mouvement synthetique a deux axes et
                # vitesse variable (27 deg d'amplitude) : 8 images-cles donnent
                # 0,24 deg d'erreur max, 16 en donnent 0,06. Inutile d'en
                # echantillonner davantage — le facteur limitant est le bruit
                # des estimateurs, pas l'interpolation.
                _cles = self.verticales_par_image_cle()
                _nf = int(getattr(CoordinateTransformer, "_n_frames_video", 0)) or len(kpts)
                # ⚠️ OPT-IN, pas defaut. Ecrite le 2026-09-09, validee sur
                # verite terrain SYNTHETIQUE (0,24 deg d'erreur sur 8 images-cles
                # pour un mouvement de 27 deg a deux axes) mais PAS sur une vraie
                # video. Le seul essai qui l'exerce — `titia_cycling_insitu`,
                # camera a la main — donne un tronc a 43,11 deg contre 35 sans
                # elle, soit 8 deg de sur-correction : la cycliste ressort plus
                # droite qu'elle ne l'est, ce que Florian a vu a l'oeil.
                # Piste a instruire : en exterieur, MoGe mesure la NORMALE DE LA
                # ROUTE et GeoCalib mesure la GRAVITE. Sur une pente les deux
                # divergent legitimement, et leur moyenne n'est ni l'une ni
                # l'autre. Il faudra peut-etre preferer GeoCalib seul dehors.
                # `MOGE_WF_MOBILE=1` la reactive pour la mettre au point.
                _mobile_on = bool(int(_os_wf.environ.get("MOGE_WF_MOBILE", "0")))
                _seuil_mob = float(_os_wf.environ.get("MOGE_WF_SEUIL_MOBILE_DEG", "2.5"))
                _disp = 0.0
                if len(_cles) >= 3:
                    _V = np.array([c[1] for c in _cles])
                    _m = _V.mean(axis=0); _m /= (np.linalg.norm(_m) or 1.0)
                    _disp = float(np.degrees(np.arccos(
                        np.clip(_V @ _m, -1.0, 1.0))).max())
                print(f"  [world frame] images-cles retenues par le consensus : "
                      f"{len(_cles)} | dispersion {_disp:.2f}° | seuil mobile "
                      f"{_seuil_mob:.1f}°")
                if _mobile_on and len(_cles) >= 3 and _disp > _seuil_mob:
                    _serie = self.interpoler_verticales(_cles, len(kpts))
                    if _serie is not None:
                        _y0 = float(np.nanmin(kpts[..., 1])) if kpts.size else 0.0
                        _cible = np.array([0.0, 1.0, 0.0])
                        for _i in range(len(kpts)):
                            _Ri = self.rotation_align(_serie[_i], _cible)
                            kpts[_i] = kpts[_i] @ _Ri.T
                            if jc is not None and _i < len(jc) and jc[_i] is not None:
                                jc[_i] = jc[_i] @ _Ri.T
                        if kpts.size:
                            _dy = _y0 - float(np.nanmin(kpts[..., 1]))
                            if np.isfinite(_dy) and abs(_dy) > 1e-6:
                                kpts[..., 1] += _dy
                                if jc is not None:
                                    jc[..., 1] += _dy
                        self._last_world_frame_series = [
                            self.rotation_align(_serie[_i2], _cible)
                            for _i2 in range(len(kpts))]
                        self._last_world_frame_dy_m = float(_dy) if np.isfinite(_dy) else 0.0
                        _incl = float(np.degrees(np.arccos(np.clip(
                            np.abs(_serie[:, 1]), -1.0, 1.0))).mean())
                        print(f"  [world frame] CAMERA MOBILE : dispersion "
                              f"{_disp:.2f}° > {_seuil_mob:.1f}° → verticale "
                              f"interpolee sur {len(_cles)} images-cles, "
                              f"inclinaison moyenne {_incl:.2f}°, rotation PAR IMAGE")
                        self._last_floor_angle_deg = _incl
                        # ⚠️ Drapeau DEDIE, pas `_world_frame = False` : ce
                        # dernier fait retomber sur la chaine historique, qui
                        # appliquerait sa propre rotation PAR-DESSUS celle-ci.
                        _deja_redresse = True
                    else:
                        _ups = list(getattr(CoordinateTransformer, "_ups_cam_m2s", []))
                        _ups = [u for _, u in _ups]
                else:
                    _ups = [u for _, u in
                            (getattr(CoordinateTransformer, "_ups_cam_m2s", []) or [])]
                if _deja_redresse:
                    # Deja redresse image par image : ne rien appliquer de plus.
                    # Sans ce test, le bloc global ci-dessous ajoutait SA
                    # rotation par-dessus la mienne — double redressement,
                    # visible au journal du 2026-09-09 : « rotation PAR IMAGE »
                    # suivi de « rotation GLOBALE » sur le meme passage.
                    pass
                elif _ups:
                    # CHEMIN FIDELE : vraies normales par image, agregation
                    # robuste avec rejet a 8 deg, conversion de repere, rotation
                    # globale unique. C est la recette Mesh2Sim complete.
                    _u, _ng, _nt, _spread = self.aggregate_ups_m2s(_ups)
                    _mg = self.cam_up_to_world_m2s(_u)
                    _gc = getattr(CoordinateTransformer, "_gc_up_world_m2s", None)

                    # CONSENSUS GeoCalib ⊕ MoGe — leur regle, mot pour mot :
                    # angle entre les deux, au-dela de 8 deg on ne fait pas
                    # confiance, sinon la verticale est la somme normalisee.
                    # Une seule source disponible : on la prend telle quelle
                    # (elle vaut 0,7-2,7 deg sur leur banc de 27 cameras).
                    if _gc is not None:
                        _ang = float(np.degrees(np.arccos(
                            np.clip(float(_mg @ _gc), -1.0, 1.0))))
                        if _ang > 8.0:
                            print(f"  [world frame] GeoCalib⊕MoGe EN DESACCORD "
                                  f"({_ang:.1f}° > 8) → verticale non fiable, "
                                  f"repli sur la mediane des angles.")
                            _world_frame = False
                            _up_w = _mg
                        else:
                            _up_w = _gc + _mg
                            _up_w = _up_w / (np.linalg.norm(_up_w) or 1.0)
                            print(f"  [world frame] GeoCalib⊕MoGe d accord "
                                  f"({_ang:.1f}°) → verticale de consensus")
                    else:
                        _up_w = _mg
                        print("  [world frame] GeoCalib indisponible → MoGe seul")

                    # Le consensus est exprime dans le monde de Mesh2Sim ; nos
                    # points sont dans le notre. UNE conversion, ici.
                    _up_w = self.up_m2s_vers_opensim(_up_w)
                    _R = self.rotation_align(_up_w, np.array([0.0, 1.0, 0.0]))
                    _incl = float(np.degrees(np.arccos(np.clip(_up_w[1], -1.0, 1.0))))
                    if _world_frame:
                        print(f"  [world frame] {_ng}/{_nt} images gardees, "
                              f"etendue {_spread:.2f}° | up monde = "
                              f"[{_up_w[0]:+.3f},{_up_w[1]:+.3f},{_up_w[2]:+.3f}]"
                              f" | inclinaison {_incl:.2f}° | rotation GLOBALE")
                        # Reposer la scene apres la rotation : elle est globale,
                        # donc centree sur l origine CAMERA, et elle deplace donc
                        # le sujet en hauteur. On rend au point le plus bas de la
                        # sequence la hauteur qu il avait avant. C est une
                        # TRANSLATION, elle ne change aucun angle. Chez Mesh2Sim
                        # ce role est tenu par leur etage ground_anchor.
                        #
                        # ⚠️ Ne PAS lui attribuer l offset de ~90 cm que le sol
                        # stable signale sur les squats. J avais fait ce lien, la
                        # mesure le dement : le temoin `--no_world_frame` donne
                        # deja -85,7 cm avec un refus a 86,5 cm, contre -94,8 avec
                        # le repere monde. L offset PREEXISTE ; la rotation n en
                        # ajoute que ~9 cm. Ici le recalage ne vaut que 5,7 cm.
                        # Le refus du sol stable sur les squats est un sujet a
                        # part, ouvert au registre.
                        _y0 = float(np.nanmin(kpts[..., 1])) if kpts.size else 0.0
                        kpts = (kpts.reshape(-1, 3) @ _R.T).reshape(kpts.shape)
                        if jc is not None:
                            jc = (jc.reshape(-1, 3) @ _R.T).reshape(jc.shape)
                        if kpts.size:
                            _dy = _y0 - float(np.nanmin(kpts[..., 1]))
                            if np.isfinite(_dy) and abs(_dy) > 1e-6:
                                kpts[..., 1] += _dy
                                if jc is not None:
                                    jc[..., 1] += _dy
                                print(f"  [world frame] scene reposee : {_dy*100:+.1f} cm "
                                      f"en Y (la rotation globale l avait soulevee)")
                        self._last_world_frame_R = _R
                        self._last_world_frame_dy_m = float(_dy) if np.isfinite(_dy) else 0.0
                        self._last_world_frame_tilt_deg = _incl
                        # ⚠️ NE PAS remplir `_last_floor_angle_deg` ici : il est
                        # rejoue sur le mesh comme une rotation autour du
                        # BASSIN, alors qu'on vient d'appliquer une rotation
                        # GLOBALE aux kpts. C'est ce qui decouplait le mesh de
                        # l'anatomical (constate sur Titia le 2026-09-09).
                        self._last_floor_angle_deg = None
                else:
                    # REPLI : aucune normale collectee (angles fournis a la main,
                    # ou estimation mono-image). On reconstitue depuis les angles
                    # bruts : ca teste la conversion de repere et la rotation
                    # globale, mais ni l orientation geometrique ni le rejet.
                    _ls = float(_os_wf.environ.get("MOGE_LEAN_SCALE", "0.5")) or 1.0
                    _p_raw, _r_raw = _pitch / _ls, -_roll / _ls
                    kpts, jc = self._apply_world_frame_leveling(kpts, jc, _p_raw, _r_raw)
                    self._last_floor_angle_deg = _p_raw
                    self._last_roll_angle_deg = _r_raw
            # Chaine historique : le defaut quand le repere monde est coupe, ET
            # le repli quand il vient de se refuser lui-meme faute de consensus.
            # Sans ce second cas, un refus laisserait la scene NON redressee.
            if not _world_frame and not _deja_redresse:
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
        self._last_stable_floor = None
        # Offset de mise au sol partage entre les appels successifs a
        # apply_pipeline_to_verts (mesh / kpts / jcoords).
        self._replay_constant_offset_m = None
        self._last_penetration_clamp_m = None  # (N,) per-frame safety-net shift
        if stable_floor:
            # SOL STABLE (portage Mesh2Sim). Exclusif des trois autres modes :
            # contact_anchor, align_to_ground et le clamp per-frame sont
            # precisement ce qu'il remplace. Le comportement historique reste
            # accessible en laissant stable_floor=False.
            from .ground_plane import apply_stable_floor, stable_floor_transform
            if plantar_indices is not None and jc is not None and len(plantar_indices):
                pidx = np.asarray(plantar_indices, dtype=int)
                plantar = jc[:, pidx, :]
                # `plantar_indices` est attendu ordonne pied droit puis pied
                # gauche (SOLE_s1..s7_r puis _l), d'ou la coupe au milieu.
                split = len(pidx) // 2 if len(pidx) >= 2 else None
                src = f"{len(pidx)} points plantaires"
            else:
                plantar = kpts[:, _FOOT_INDICES, :]
                split = 2
                src = "4 keypoints cutanes (pas de points plantaires fournis)"
            sf = stable_floor_transform(plantar, float(fps), per_foot_split=split,
                                        drift_model=stable_floor_drift,
                                        min_span_travel_m=2.0)
            print(f"  [stable floor] source : {src} | rotation "
                  f"{'OUI' if sf.rotation_applied else 'non'} "
                  f"(inclinaison {sf.fit.tilt_deg:.2f}deg, {sf.fit.reason or 'ok'}) | "
                  f"derive {'OUI' if sf.drift_applied else 'non'} "
                  f"({sf.drift_m_per_s*100:+.2f} cm/s, appui "
                  f"{sf.stance_coverage*100:.0f}%) | offset {sf.offset_m*100:+.1f} cm"
                  + (f" | {sf.notes}" if sf.notes else ""))
            kpts = apply_stable_floor(kpts, sf, float(fps))
            if jc is not None:
                jc = apply_stable_floor(jc, sf, float(fps))
            self._last_stable_floor = sf
            self._last_stable_floor_fps = float(fps)
        elif contact_anchor:
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
            # ⚠️ Points de SEMELLE quand ils existent, articulations sinon.
            # `_FOOT_INDICES` designe les CENTRES ARTICULAIRES talon et gros
            # orteil, qui flottent quelques centimetres au-dessus de la semelle :
            # se caler dessus met le sujet en l'air d'autant. Le sol stable
            # utilise deja les points plantaires ; cette branche et le clamp
            # ci-dessous ne le faisaient pas (constat du 2026-09-09).
            def _pieds(i):
                if plantar_indices is not None and jc is not None and len(plantar_indices):
                    return jc[i, np.asarray(plantar_indices, dtype=int)]
                return kpts[i, _FOOT_INDICES]
            # ⚠️ CALIBRE SUR TOUT L'ESSAI, PAS SUR LES 20 PREMIERES IMAGES.
            # L'ancienne fenetre supposait que le point le plus bas du geste
            # arrive au debut. C'est faux des qu'il y a un CYCLE : en pedalage,
            # le point mort bas revient toutes les 0,7 s et rien ne dit qu'il
            # tombe dans les 20 premieres images. Mesure sur
            # `output_20260909_200817_bikefit_demo` (2026-09-09) : 16 images sur
            # 241 descendaient a -3,5 cm, le pied passant sous le sol dans le
            # visionneur. Florian : « parfois leurs pieds vont en dessous du
            # sol ; trouve le min sur tout l'essai et remonte d'un offset fixe
            # avec un peu de marge ».
            #
            # L'offset reste CONSTANT sur l'essai — c'est ce qui garantit qu'il
            # ne deforme rien : une translation verticale uniforme ne change
            # aucun angle articulaire ni aucune amplitude. Et il est rejoue tel
            # quel sur le mesh (`_last_constant_offset_m`), donc peau, squelette
            # et TRC restent alignes au millimetre.
            #
            # Cette branche n'est atteinte que quand le SOL STABLE est inactif,
            # c'est-a-dire en pratique le cyclisme et la quadrupedie : les
            # gestes ou les pieds ne touchent aucun sol et ou personne d'autre
            # ne garantit qu'ils restent au-dessus.
            _mins = []
            for i in range(kpts.shape[0]):
                foot = _pieds(i)
                if not np.any(np.isnan(foot)):
                    _mins.append(float(np.min(foot[:, 1])))
            calib_min_y = _mins
            if calib_min_y:
                _plus_bas = float(np.min(calib_min_y))
                _debut = float(np.min(calib_min_y[:min(20, len(calib_min_y))]))
                # Marge : le point le plus bas se retrouve a `ground_margin_m`
                # au-dessus du sol plutot qu'exactement dessus. Sans elle, la
                # moindre interpolation d'affichage repasse sous zero.
                constant_offset = _plus_bas - ground_margin_m
                print(f"  [floor lean] constant ground shift Y -= "
                      f"{constant_offset:.3f} m (point le plus bas de l'essai "
                      f"{_plus_bas*100:+.1f} cm sur {len(calib_min_y)} images, "
                      f"marge {ground_margin_m*100:.0f} cm ; les 20 premieres "
                      f"images seules auraient donne {_debut*100:+.1f} cm, soit "
                      f"{(_debut-_plus_bas)*100:+.1f} cm d'ecart)")
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
                    foot = _pieds(i)
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
                # Verticale de la translation camera, rejouee comme transform().
                if (self._last_stationary_cam_t_y_m is not None
                        and i < len(self._last_stationary_cam_t_y_m)):
                    w[:, 1] += self._last_stationary_cam_t_y_m[i]
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
            # ── REPERE MONDE : rejouer la MEME transformation que les kpts ──
            # ⚠️ Defaut trouve le 2026-09-09, signale par Florian sur Titia :
            # « le mesh et l'anatomical sont decouples ». En mode repere monde,
            # les points articulaires subissent une rotation GLOBALE autour de
            # l'origine camera, tandis que ce bloc rejouait la chaine
            # historique — une rotation autour du BASSIN, d'angle
            # `_last_floor_angle_deg`. Deux transformations differentes sur les
            # memes donnees : le mesh et le squelette divergeaient
            # necessairement, et d'autant plus que l'inclinaison etait grande.
            # `_last_world_frame_R` etait d'ailleurs stocke depuis le portage et
            # JAMAIS relu.
            _Rw = getattr(self, "_last_world_frame_R", None)
            _serie_w = getattr(self, "_last_world_frame_series", None)
            if _serie_w is not None and i < len(_serie_w):
                w = w @ _serie_w[i].T
                w[:, 1] += getattr(self, "_last_world_frame_dy_m", 0.0)
                pre_ground.append(w)
                continue
            if _Rw is not None:
                w = w @ _Rw.T
                w[:, 1] += getattr(self, "_last_world_frame_dy_m", 0.0)
                pre_ground.append(w)
                continue
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
        _sf = getattr(self, "_last_stable_floor", None)
        if _sf is not None:
            # Sol stable : la MEME transformation que pour les kpts, rejouee a
            # l'identique image par image. Rigide par image, donc le mesh reste
            # exactement colle au squelette (ecart nul par construction).
            for i, w in enumerate(pre_ground):
                if w is None:
                    continue
                if _sf.rotation_applied:
                    w = (w - _sf.pivot) @ _sf.R.T + _sf.pivot
                dy = _sf.offset_m
                if _sf.drift_m_per_s:
                    dy += _sf.drift_m_per_s / float(self._last_stable_floor_fps) \
                          * (i - _sf.t0_frame)
                if _sf.stance_shift_m is not None and i < len(_sf.stance_shift_m):
                    dy += float(_sf.stance_shift_m[i])
                w = w.copy()
                w[:, 1] -= dy
                pre_ground[i] = w
        elif ground_offset_mode == "per_frame":
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
            elif self._replay_constant_offset_m is not None:
                # Le mesh, les keypoints et les centres articulaires arrivent en
                # TROIS appels separes. Sans partage, chacun recalculait son
                # offset sur SES propres donnees — donc trois translations
                # differentes (mesure : 0,673 / 0,702 / 0,000 m sur un bird dog),
                # d'ou des spheres articulaires flottant au-dessus de la peau et
                # un anatomical passant sous le sol. Le premier appel fait
                # reference, les suivants le rejouent.
                constant_offset = self._replay_constant_offset_m
                print(f"  [apply_pipeline_to_verts] constant_from_calib: "
                      f"Y -= {constant_offset:.3f} m (offset partage)")
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
                self._replay_constant_offset_m = constant_offset
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

        # Estimation par SOMME DE SEGMENTS quand les jcoords sont disponibles.
        # La mesure tete-pied ci-dessus suppose un sujet qui s'etend au moins une
        # fois dans le clip — le percentile 95 attrape cet instant sur un squat ou
        # un sit-to-stand. Mais un CYCLISTE ne se redresse jamais : hanches et
        # genoux flechis, tronc penche, sa distance tete-pied vaut bien moins que
        # sa taille, et la forcer a subject_height dilate tout le squelette
        # (mesure sur un cyclisme in-situ : +22 %, femur a 0,599 m au lieu de 0,49).
        # La somme de segments ne depend pas de la posture.
        if jc is not None and jc.shape[1] > 126:
            seg = []
            for a, b in _STATURE_CHAIN:
                d = np.linalg.norm(jc[:, a] - jc[:, b], axis=1)
                d = d[np.isfinite(d)]
                if d.size:
                    seg.append(float(np.median(d)))   # os rigide -> mediane
            if len(seg) == len(_STATURE_CHAIN):
                stature = sum(seg) * _STATURE_CHAIN_TO_HEIGHT
                if stature > 0.5:
                    return self.subject_height / stature

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
        CoordinateTransformer._n_frames_video = int(total)
        sample_idx = np.linspace(margin, total - margin - 1, n_samples).astype(int)
        sample_idx = np.unique(sample_idx)

        # SECOND ESTIMATEUR (GeoCalib) sur LES MEMES images. Chez Mesh2Sim, MoGe
        # n est jamais seul : les deux doivent s accorder a moins de 8 deg, sinon
        # la verticale est jugee non fiable. Mesure du 2026-09-08 sur
        # `Sprint_start` : MoGe donne des pitchs de 24 a 81 deg selon l image et
        # son agregation s effondre a 1 image sur 8, quand GeoCalib garde 7/7
        # avec 0,43 deg d etendue et 1,24 deg d inclinaison. C est exactement le
        # role d arbitre que joue GeoCalib chez eux.
        import os as _os_gc
        CoordinateTransformer._gc_up_world_m2s = None
        if bool(int(_os_gc.environ.get("MOGE_WORLD_FRAME", "1"))):
            try:
                CoordinateTransformer._gc_up_world_m2s = (
                    CoordinateTransformer.geocalib_up_world_m2s(
                        video_path, [int(i) for i in sample_idx]))
            except Exception as _e_gc:
                print(f"  [geocalib] indisponible ({str(_e_gc)[:70]})")

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

            # MODE REPERE MONDE : on retient AUSSI la normale brute de cette
            # image, en repere camera. C est le SEUL endroit du pipeline ou la
            # grille de points MoGe existe ; plus loin il ne reste que des angles
            # deja reduits, dont on ne peut reconstituer ni l orientation
            # geometrique ni le rejet d image.
            import os as _os_wf2
            # La COLLECTE reste active meme quand l'application est coupee :
            # elle ne coute rien et alimente le diagnostic du consensus.
            if bool(int(_os_wf2.environ.get("MOGE_WORLD_FRAME", "1"))):
                try:
                    _u = CoordinateTransformer.floor_up_from_points_m2s(
                        pts, mask, person_bbox=person_bbox, orig_hw=orig_hw)
                    if _u is not None:
                        CoordinateTransformer._ups_cam_m2s.append((int(idx), _u))
                except Exception:
                    pass

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

    # ------------------------------------------------------------------
    # MODE REPERE MONDE (MOGE_WORLD_FRAME=1) — portage de la recette Mesh2Sim
    # ------------------------------------------------------------------
    @staticmethod
    def world_up_from_moge_angles(pitch_deg: float, roll_deg: float) -> np.ndarray:
        """Vecteur "haut" du monde, reconstruit depuis les angles MoGe bruts.

        POURQUOI CETTE CONVERSION EXISTE. MoGe rend ses points dans la convention
        OpenCV : x a droite, y vers le BAS, z vers l avant. Le monde OpenSim est
        x a droite, y vers le HAUT, z vers l arriere. Le passage de l un a
        l autre est (x, -y, -z) — une ROTATION de 180 degres autour de x.

        LE PIEGE, documente par Mesh2Sim et paye 22 degres d erreur chez eux :
        nier le vecteur ENTIER (-x, -y, -z) pour le retourner n est pas la meme
        chose. C est une REFLEXION, elle inverse l axe gauche-droite, donc le
        SIGNE DU ROULIS. Leurs deux estimateurs tombaient alors "d accord" sur un
        roulis faux, et leur mode consensus ne pouvait structurellement pas
        fonctionner. Se tromper ici penche le sujet de plusieurs degres sans
        produire la moindre erreur visible.

        Notre chaine historique ne fait AUCUNE conversion : elle calcule le pitch
        et le roll directement depuis la normale en repere OpenCV, ou +y pointe
        vers le bas. Deux compensations empiriques la rattrapent en pratique — le
        facteur LEAN_SCALE et la negation du roulis — et le sol reconstruit
        ressort horizontal a plus ou moins 1 degre sur nos courses. Ce mode-ci ne
        les utilise pas : il convertit le repere, puis applique l angle entier.

        Args:
            pitch_deg, roll_deg: angles BRUTS (non multiplies par LEAN_SCALE),
                tels que definis par la chaine MoGe : pitch = atan2(n_z, n_y),
                roll = atan2(n_x, n_y), normale en repere OpenCV avec n_y > 0.

        Returns:
            (3,) vecteur unitaire "haut" en repere monde (y > 0).
        """
        ny = 1.0
        n_cv = np.array([ny * np.tan(np.radians(roll_deg)),
                         ny,
                         ny * np.tan(np.radians(pitch_deg))], dtype=np.float64)
        n_cv /= (np.linalg.norm(n_cv) or 1.0)
        # OpenCV -> monde : rotation de 180 deg autour de x, PAS une negation.
        n_w = n_cv * np.array([1.0, -1.0, -1.0])
        if n_w[1] < 0:          # normale de signe libre : on veut "vers le haut"
            n_w = -n_w
        return n_w / (np.linalg.norm(n_w) or 1.0)

    @staticmethod
    def verticales_par_image_cle(max_desaccord_deg: float = 8.0):
        """Consensus GeoCalib ⊕ MoGe image-clé PAR image-clé.

        La version globale agrège d'abord et confronte ensuite : une seule
        verticale pour tout l'essai. C'est juste quand la caméra ne bouge pas,
        et faux dès qu'elle bouge — la verticale n'est alors pas la même à
        l'image 10 et à l'image 150, et refuser la correction (ce que fait le
        mode global) revient à ne rien redresser du tout.

        Ici on applique LEUR règle des 8 degrés à chaque image-clé, et une
        image-clé où les deux divergent est simplement ÉCARTÉE : on interpole
        par-dessus au lieu de faire tomber tout l'essai.

        Returns:
            [(idx_image, up_monde)] trié par indice, ou [] si rien d'exploitable.
        """
        gc = dict(getattr(CoordinateTransformer, "_gc_ups_world_m2s", []) or [])
        mg_cam = dict(getattr(CoordinateTransformer, "_ups_cam_m2s", []) or [])
        if not gc and not mg_cam:
            return []
        mg = {i: CoordinateTransformer.cam_up_to_world_m2s(u)
              for i, u in mg_cam.items()}
        out = []
        _diag = []
        for i in sorted(set(gc) | set(mg)):
            a, b = gc.get(i), mg.get(i)
            if a is not None and b is not None:
                ang = float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))
                _diag.append((i, ang))
                if ang > max_desaccord_deg:
                    continue                      # image-clé écartée
                v = a + b
            else:
                v = a if a is not None else b     # une seule source disponible
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                # Meme passerelle que le chemin global : les cles sont
                # rendues dans NOTRE repere, pretes a etre appliquees.
                out.append((int(i), CoordinateTransformer.up_m2s_vers_opensim(v / n)))
        if _diag:
            print("  [world frame] desaccord GeoCalib/MoGe par image-cle : "
                  + " ".join(f"{i}:{a:.1f}°" for i, a in _diag))
        return out

    @staticmethod
    def interpoler_verticales(cles, n_images: int):
        """Verticale par image, interpolée entre les images-clés (slerp).

        ⚠️ On interpole des ORIENTATIONS, pas des angles. Une moyenne linéaire
        de pitch et de roll dérape dès que les deux varient ensemble ; le slerp
        parcourt le grand cercle entre deux directions unitaires, ce qui est la
        trajectoire naturelle d'une caméra qui pivote.

        Hors de l'intervalle des images-clés, on prolonge par la valeur du bord
        plutôt que d'extrapoler : une caméra fait n'importe quoi avant et après
        la séquence utile, et extrapoler y inventerait un mouvement.
        """
        if not cles or n_images <= 0:
            return None
        idx = np.array([c[0] for c in cles], dtype=float)
        V = np.array([c[1] for c in cles], dtype=np.float64)
        if len(cles) == 1:
            return np.repeat(V[0][None, :], n_images, axis=0)
        t = np.arange(n_images, dtype=float)
        sortie = np.empty((n_images, 3), dtype=np.float64)
        pos = np.searchsorted(idx, t, side="right") - 1
        pos = np.clip(pos, 0, len(idx) - 2)
        for k in range(n_images):
            j = int(pos[k])
            a, b = V[j], V[j + 1]
            if t[k] <= idx[0]:
                sortie[k] = V[0]; continue
            if t[k] >= idx[-1]:
                sortie[k] = V[-1]; continue
            u = (t[k] - idx[j]) / max(idx[j + 1] - idx[j], 1e-9)
            cos = float(np.clip(a @ b, -1.0, 1.0))
            om = float(np.arccos(cos))
            if om < 1e-6:
                v = (1.0 - u) * a + u * b
            else:
                so = np.sin(om)
                v = (np.sin((1.0 - u) * om) / so) * a + (np.sin(u * om) / so) * b
            n = float(np.linalg.norm(v))
            sortie[k] = v / n if n > 1e-9 else a
        return sortie

    @staticmethod
    def geocalib_up_world_m2s(video_path: str, frames, camera_model: str = "pinhole"):
        """Verticale de la scene par GeoCalib — portage de leur etage mono.

        Reproduit `stages/calibration_mono_geocalib/.../estimate.py` : `up` vaut
        `-gravity.vec3d`, estime sur plusieurs images, puis la MEME agregation
        que pour MoGe (alignement de signe, moyenne, rejet au-dela de 8 deg,
        re-moyenne).

        ⚠️ LE DETAIL QUI DECIDE DE TOUT, et qu ils ont paye 22 deg d erreur pour
        trouver : `-gravity.vec3d` tombe deja en y haut / z arriere, mais avec
        l axe X **INVERSE** par rapport a la calibration, a MoGe et a la colonne
        (roulis de signe oppose mesure sur deux cameras : +3,35 contre -3,8 deg,
        +10,96 contre -10,7). On inverse donc x, et RIEN d autre — surtout pas
        une negation du vecteur entier, qui serait une reflexion. Sans cette
        correction les deux sources tombent « d accord » sur un roulis faux et le
        consensus ne peut structurellement pas fonctionner.

        Returns:
            (3,) vecteur unitaire "haut" en repere MONDE, ou None.
        """
        import os as _os_g, tempfile as _tf
        import cv2 as _cv
        import torch as _t
        from geocalib import GeoCalib as _GC

        _t.set_grad_enabled(False)
        dev = "cuda" if _t.cuda.is_available() else "cpu"
        model = _GC().to(dev)
        ups, vus = [], []
        for f in frames:
            cap = _cv.VideoCapture(video_path)
            cap.set(_cv.CAP_PROP_POS_FRAMES, int(f))
            ok, bgr = cap.read()
            cap.release()
            if not ok or bgr is None:
                continue
            tmp = _tf.mktemp(suffix=".png")
            try:
                _cv.imwrite(tmp, bgr)
                img = model.load_image(tmp).to(dev)
            finally:
                if _os_g.path.exists(tmp):
                    _os_g.unlink(tmp)
            r = model.calibrate(img, camera_model=camera_model)
            u = -r["gravity"].vec3d[0].cpu().numpy()
            n = float(np.linalg.norm(u))
            if n > 1e-9:
                ups.append(u / n); vus.append(f)
        if not ups:
            return None
        # Memorise le vecteur de CHAQUE image-cle, en repere monde : c'est ce
        # qui permet de suivre une camera qui bouge au lieu de tout moyenner.
        CoordinateTransformer._gc_ups_world_m2s = [
            (int(f), (u * np.array([-1.0, 1.0, 1.0]))
                     / (np.linalg.norm(u * np.array([-1.0, 1.0, 1.0])) or 1.0))
            for f, u in zip(vus, ups)]
        V = np.array(ups)
        V[V @ V[0] < 0] *= -1
        mean = V.mean(axis=0)
        mean /= (np.linalg.norm(mean) or 1.0)
        dev_deg = np.degrees(np.arccos(np.clip(V @ mean, -1, 1)))
        keep = dev_deg <= 8.0
        if keep.sum() >= 1:
            mean = V[keep].mean(axis=0)
            mean /= (np.linalg.norm(mean) or 1.0)
        # X inverse — voir l avertissement du docstring.
        out = mean * np.array([-1.0, 1.0, 1.0])
        out /= (np.linalg.norm(out) or 1.0)
        print(f"  [geocalib] {int(keep.sum())}/{len(V)} images gardees, etendue "
              f"{float(dev_deg[keep].max()) if keep.any() else 0.0:.2f}° | up monde "
              f"= [{out[0]:+.3f},{out[1]:+.3f},{out[2]:+.3f}] | inclinaison "
              f"{np.degrees(np.arccos(np.clip(abs(out[1]), -1, 1))):.2f}°")
        return out

    @staticmethod
    def floor_up_from_points_m2s(points, mask, person_bbox=None, orig_hw=None,
                                 floor_frac: float = 0.30, n_samples: int = 6000):
        """Normale du sol depuis une grille de points MoGe — portage fidele Mesh2Sim.

        Reproduit `stages/orientation_moge/.../estimate.py:floor_up_from_points`.
        Renvoie le vecteur "haut" BRUT en repere camera (OpenCV), sans facteur
        d echelle, ou None.

        DEUX DIFFERENCES AVEC NOTRE CHAINE HISTORIQUE, et la seconde est celle qui
        compte :

        1. Parametres : bande de sol a 30 pour cent des lignes (nous : 25),
           6000 points echantillonnes (nous : 4000), RANSAC a 300 iterations
           (nous : 200). Reglages, sans consequence de principe.

        2. ORIENTATION DE LA NORMALE. Une normale de plan a un signe libre ; il
           faut le fixer. Eux le font GEOMETRIQUEMENT : la normale "haut" doit
           pointer du sol VERS la camera, donc son produit scalaire avec le point
           de sol moyen doit etre negatif. Nous forcons `normal[1] > 0`, un test
           sur un AXE — dans un repere ou +y pointe justement vers le BAS. Le
           critere geometrique ne depend d aucune convention d axe et ne peut pas
           se tromper de repere ; le notre le peut, et c est exactement la famille
           d erreur qui leur a coute 22 degres.
        """
        points = np.asarray(points)
        if points.ndim != 3:
            return None
        H, W = points.shape[:2]
        valid = np.asarray(mask).astype(bool)
        floor_row_min = None
        if person_bbox is not None and orig_hw is not None:
            oh, ow = orig_hw
            x1, y1, x2, y2 = person_bbox
            gx1, gy1 = int(x1 / ow * W), int(y1 / oh * H)
            gx2, gy2 = int(x2 / ow * W), int(y2 / oh * H)
            mx = max(1, int((gx2 - gx1) * 0.10))
            my = max(1, int((gy2 - gy1) * 0.10))
            valid[max(0, gy1 - my):min(H, gy2 + my) + 1,
                  max(0, gx1 - mx):min(W, gx2 + mx) + 1] = False
            below = max(1, int((gy2 - gy1) * 0.05))
            floor_row_min = min(H - 1, gy2 + below)
            band = np.zeros_like(valid)
            band[floor_row_min:, :] = True
            valid = valid & band

        pv = points.reshape(-1, 3).astype(np.float64)[valid.reshape(-1)]
        if len(pv) < 50:
            return None
        if floor_row_min is not None:
            floor_pts = pv
        else:
            # Y-DOWN (OpenCV) : le sol est du cote des GRANDS Y/Z (bas de l image)
            yz = pv[:, 1] / pv[:, 2]
            floor_pts = pv[yz >= np.percentile(yz, (1.0 - floor_frac) * 100)]
        if len(floor_pts) < 20:
            return None
        if len(floor_pts) > n_samples:
            floor_pts = floor_pts[np.random.default_rng(0).choice(
                len(floor_pts), n_samples, replace=False)]

        normal, _ = CoordinateTransformer._ransac_plane_normal(
            floor_pts, n_iter=300, dist_thresh=0.05, min_inliers=50)
        if normal is None:
            centroid = floor_pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(floor_pts - centroid, full_matrices=False)
            normal = Vt[-1]
        normal = normal / (np.linalg.norm(normal) or 1.0)
        # ORIENTATION GEOMETRIQUE : du sol vers la camera (l origine).
        if float(normal @ floor_pts.mean(axis=0)) > 0:
            normal = -normal
        return normal

    @staticmethod
    def aggregate_ups_m2s(ups, max_dev_deg: float = 8.0):
        """Direction mediane robuste sur plusieurs images — portage Mesh2Sim.

        Reproduit `_aggregate_ups` : on aligne tous les vecteurs en signe sur le
        premier, on prend la moyenne unitaire, on ECARTE les images a plus de
        ``max_dev_deg`` de cette moyenne, puis on remoyenne. L etendue angulaire
        restante est un signal de stabilite : une seule image aberrante est
        rejetee au lieu de contaminer l estimation.

        Notre chaine historique moyenne 8 images sans ecarter les aberrantes.

        Returns:
            (up_moyen, n_gardees, n_total, etendue_deg) ou (None, 0, 0, 0.0).
        """
        ups = [np.asarray(u, dtype=np.float64) for u in ups if u is not None]
        if not ups:
            return None, 0, 0, 0.0
        V = np.array([u / (np.linalg.norm(u) or 1.0) for u in ups])
        V[V @ V[0] < 0] *= -1                      # alignement de signe
        moy = V.mean(axis=0); moy /= (np.linalg.norm(moy) or 1.0)
        dev = np.degrees(np.arccos(np.clip(V @ moy, -1, 1)))
        garde = dev <= max_dev_deg
        if garde.sum() >= 1:
            moy = V[garde].mean(axis=0); moy /= (np.linalg.norm(moy) or 1.0)
            dev_g = np.degrees(np.arccos(np.clip(V[garde] @ moy, -1, 1)))
        else:
            dev_g = dev
        return moy, int(garde.sum()), int(len(V)), float(dev_g.max() if len(dev_g) else 0.0)

    @staticmethod
    def cam_up_to_world_m2s(up_cam: np.ndarray) -> np.ndarray:
        """Repere camera OpenCV -> repere monde. Portage de `run_mono_view.py:418-424`.

        (x, -y, -z) est une ROTATION de 180 degres autour de x. Nier le vecteur
        entier serait une REFLEXION : elle inverse l axe gauche-droite, donc le
        signe du ROULIS. C est le bug qu ils documentent, 22 degres d erreur
        cumulee, et leur mode consensus ne pouvait pas fonctionner tant qu il
        etait la.
        """
        u = np.asarray(up_cam, dtype=np.float64)
        u = u * np.array([1.0, -1.0, -1.0])
        if u[1] < 0:
            u = -u
        return u / (np.linalg.norm(u) or 1.0)

    @staticmethod
    def up_m2s_vers_opensim(m: np.ndarray) -> np.ndarray:
        """Repere monde de Mesh2Sim -> NOTRE repere OpenSim.

        ⚠️ LA CAUSE du repere monde qui penchait le sujet au lieu de le
        redresser (2026-09-09). Mesh2Sim exprime sa verticale dans un monde
        (x droite, y haut, z arriere), obtenu depuis la camera OpenCV par
        (x, -y, -z). Notre CAMERA_TO_OPENSIM est une PERMUTATION differente :
        X_os = Z_cam, Y_os = -Y_cam, Z_os = X_cam — le monde OpenSim standard
        (x avant, y haut, z droite). J'avais porte leur conversion a la lettre,
        et elle est juste... dans LEUR monde. Nos points, eux, sont dans le
        notre. Consequence mesuree en synthetique : pour un tangage de 12 deg,
        16,91 deg d'ecart entre les deux vecteurs — le tangage etait applique
        comme un ROULIS. Sur Squat.MP4, tronc debout 10,6 deg sans correction,
        19,0 deg avec.

        La passerelle se DERIVE de CAMERA_TO_OPENSIM plutot que d'etre ecrite
        en dur : ours = C @ M @ m, avec M = diag(1,-1,-1) qui est son propre
        inverse. Verifie a 0,000 deg en synthetique.
        """
        m = np.asarray(m, dtype=np.float64)
        o = CoordinateTransformer.CAMERA_TO_OPENSIM @ (m * np.array([1.0, -1.0, -1.0]))
        if o[1] < 0:
            o = -o
        return o / (np.linalg.norm(o) or 1.0)

    @staticmethod
    def rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Rotation minimale amenant le vecteur a sur le vecteur b (Rodrigues)."""
        a = np.asarray(a, dtype=np.float64); a /= (np.linalg.norm(a) or 1.0)
        b = np.asarray(b, dtype=np.float64); b /= (np.linalg.norm(b) or 1.0)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if s < 1e-12:
            return np.eye(3) if c > 0 else -np.eye(3)
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    def _apply_world_frame_leveling(self, kpts, jc, pitch_deg, roll_deg):
        """Redresse la SCENE ENTIERE par une rotation globale unique.

        Difference de fond avec la chaine historique, qui tourne le squelette
        AUTOUR DU BASSIN : le bassin bouge d une image a l autre, donc cette
        rotation-la n est pas une transformation rigide du monde. Elle redresse
        la pose sans redresser la trajectoire. Ici on applique une rotation
        unique autour de l origine, comme Mesh2Sim (ground_anchor/mono.py:836),
        ce qui preserve la geometrie relative de tout l essai et permet ensuite
        de definir un sol constant.
        """
        up = self.up_m2s_vers_opensim(self.world_up_from_moge_angles(pitch_deg, roll_deg))
        R = self.rotation_align(up, np.array([0.0, 1.0, 0.0]))
        incl = float(np.degrees(np.arccos(np.clip(up[1], -1.0, 1.0))))
        print(f"  [world frame] up monde = [{up[0]:+.3f},{up[1]:+.3f},{up[2]:+.3f}] "
              f"| inclinaison corrigee {incl:.2f}° | rotation GLOBALE")
        kpts = (kpts.reshape(-1, 3) @ R.T).reshape(kpts.shape)
        if jc is not None:
            jc = (jc.reshape(-1, 3) @ R.T).reshape(jc.shape)
        self._last_world_frame_R = R
        self._last_world_frame_tilt_deg = incl
        return kpts, jc

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
    planted_min_s: float = 0.35,
    en_place: bool = False,
    anchor_hold_s: float = 0.0,
    shift_lowpass_hz: float = 2.0,
    max_shift_m: float = 0.0,
    continuous_contact: bool = False,
    contact_loss: str = "anchor",
    var_smooth: float = 3.0,
    var_prior: float = 0.02,
    settle_s: float = 0.12,
    max_settle_m: float = 0.12,
    ramp_s: float = 0.04,
    planted_speed_ms: float = 5.0,
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
    # DEUX JEUX DE SEUILS. Demande de Florian, 2026-09-09 : « fais un
    # anti-glissement avec des seuils differents de celui qui sert pour la
    # locomotion ». Un geste EN PLACE (squat, lever de chaise, saut vertical)
    # et une foulee ne se ressemblent en rien :
    #   * en place, le pied reste pose des secondes et ne doit PAS bouger ; une
    #     perte de contact d'un dixieme de seconde est un rate de detection,
    #     pas un envol, et jeter l'ancre pour cela revient a baptiser
    #     legitime le glissement deja accumule ;
    #   * en locomotion, un vol de 0,15 s est reel et l'ancre DOIT expirer,
    #     sinon le sujet ne progresse plus.
    # Mesure sur `outputs/SQUAT_XZ` (2026-09-09) : le pied droit reposait son
    # ancre cinq fois, la consigne oscillait entre -8,8 et +5 cm au lieu de
    # suivre une derive de 20 cm, et 3 cm seulement etaient corriges. Un
    # ancrage rigide sur la position initiale des deux pieds ramene, lui,
    # l'excursion de 21/19 a 4/3 cm : la correction est disponible, c'est
    # l'expiration des ancres qui l'empechait.
    if en_place:
        min_contact_s = max(min_contact_s, 0.25)
        max_gap_s = max(max_gap_s, 0.40)
        anchor_hold_s = max(anchor_hold_s, 0.60)
        shift_lowpass_hz = max(shift_lowpass_hz, 8.0)
        # BORNE SUR LA CORRECTION CUMULEE. Un geste en place ne demande que
        # quelques centimetres : mesure sur `outputs/SQUAT_PLACE`, 12 cm
        # suffisent pendant les squats. Au-dela, ce n'est plus du glissement.
        # Florian, 2026-09-09 : « le mec marche un pas apres, et au moment ou le
        # pied de devant se fixe tout le corps recule sur le pied arriere ».
        # C'est ce qui arrive quand un second pied se plante : il ajoute une
        # contrainte, la moyenne des deux consignes tire le corps en arriere, et
        # rien ne bornait l'ampleur (74 cm sur l'essai complet). La borne laisse
        # le geste intact et empeche la derive de fin.
        if max_shift_m <= 0:
            max_shift_m = 0.25
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
    # sont pris s'ils existent.
    # Jeu repris de l'étage ground_anchor de Mesh2Sim, rodé sur de la marche.
    #
    # ⚠️ La règle des semelles cherchait des noms de la forme `Rs1` — nos points
    # s'appellent `SOLE_s1_r`. Elle ne renvoyait donc RIEN, des deux côtés,
    # alors que le TRC porte bien 14 points plantaires (constaté 2026-09-08).
    # L'anti-glissement ne voyait que des marqueurs CUTANÉS, qui ne descendent
    # jamais sous ~3 cm du sol : le seuil de hauteur jugeait un contact sur des
    # points qui ne touchent jamais. On accepte les deux nominations.
    _base = ("CAL", "TOE", "MT5", "LMAL", "MMAL")
    side_markers = {}
    for side in ("L", "R"):
        suf = "_" + side.lower()
        idx = [name_to_idx[side + n] for n in _base if side + n in name_to_idx]
        idx += [i for n, i in name_to_idx.items()
                if n.startswith("SOLE_") and n.endswith(suf)]
        idx += [i for n, i in name_to_idx.items()
                if n.startswith(side) and len(n) > 2 and n[1] == "s" and n[2:].isdigit()]
        side_markers[side] = sorted(set(idx))
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

    # Centroide XZ de chaque pied, calcule AVANT la detection : le critere de
    # recul ci-dessous prend l'autre pied pour reference et a donc besoin des
    # deux d'emblee.
    centro = {}
    for side, idx in side_markers.items():
        if idx:
            centro[side] = np.nanmean(result[:, idx, :][:, :, [0, 2]], axis=1)

    # Contact + position XZ par côté.
    foot_xz = {}
    foot_y = {}
    contact = {}
    planted = {}
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
        # ... mais mesure PAR RAPPORT A L'AUTRE PIED, pas au bassin.
        #
        # Le critere separe l'appui du vol, et sa force est de s'affranchir du
        # bruit commun par difference. Encore faut-il que la reference soit
        # immobile quand le pied l'est. Le BASSIN ne l'est pas : en squat il
        # descend et recule, si bien que les pieds « avancent » par rapport a
        # lui alors qu'ils sont plantes. Mesure sur `outputs/SQUAT_VY`
        # (2026-09-09) : la hauteur voyait le pied au sol sur 97 a 100 % des
        # images, ce critere ramenait a 70 % et DECOUPAIT l'appui en quatre a
        # cinq phases. Or chaque nouvelle phase repose une ancre a la position
        # COURANTE du pied — la ou il a deja glisse : le glissement etait
        # valide cinq fois de suite, jusqu'a 28 cm de derive sur une phase.
        # Florian : « comprends l'anti-glissement du depart sprint, pourquoi il
        # ne bloque pas bien les pieds sur squat en XZ ».
        #
        # L'AUTRE PIED est la bonne reference. En locomotion les pieds
        # alternent : le pied qui oscille avance franchement par rapport a
        # celui qui est pose, et le glissement global, commun aux deux,
        # s'annule par difference. En squat les deux pieds sont plantes, leur
        # ecart ne bouge pas, et le critere se tait — ce qu'on veut. Il reste
        # donc actif la ou il sert (marche, course, sprint) sans mutiler les
        # gestes en place, et la marche qui SUIT les squats est preservee.
        _autre = next((o for o in centro if o != side), None)
        if walk_axis is not None and _autre is not None:
            rel = (centro[side] - centro[_autre]) @ walk_axis
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
        # DUREE de la phase d'appui en cours, image par image. Elle sert de
        # discriminant physique pour le plafond de correction (plus bas) : une
        # foulee de course dure 0,1 a 0,25 s, un pied de squat ou de lever de
        # chaise reste pose plusieurs secondes. Le premier merite la prudence,
        # le second EST la reference.
        _d = np.zeros(T)
        _i = 0
        while _i < T:
            if keep[_i]:
                _j = _i
                while _j < T and keep[_j]:
                    _j += 1
                _d[_i:_j] = (_j - _i) / max(fps, 1e-6)
                _i = _j
            else:
                _i += 1
        planted[side] = _d >= planted_min_s

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
    # ── COUVERTURE CONTINUE (MARCHE UNIQUEMENT) ──────────────────────────
    # Portage de `M2S:contact_optim.py:ensure_continuous_coverage`. En marche il
    # y a TOUJOURS un pied au sol — simple appui alternant avec double appui —
    # donc une image sans contact detecte est un RATE DE DETECTION, pas un vol.
    # Leur mesure sur P06 cam00 : couverture de 80 % seulement, et dans chaque
    # trou un pied etait demontrablement au sol et glissait de 57 a 386 mm,
    # faute d'ancre. La notre sur `outputs/GAIT_COULOIR` : 90 % de couverture,
    # cinq trous dont un de 0,40 s, et le pied droit glissait de 26 a 74 cm.
    #
    # ⚠️ MARCHE SEULEMENT. En course, au sprint, au saut et au hop unipodal le
    # vol est REEL : forcer la couverture y transformerait chaque envol en
    # appui. C'est aussi pourquoi Mesh2Sim ne l'active que sur les points
    # PLANTAIRES (`fill_contact_gaps or use_sole`) : avec des marqueurs cutanes,
    # 4 cm au-dessus de la semelle, elle degrade (2,16 -> 2,44 cMAE chez eux).
    # Nous avons les points SOLE, la condition est remplie.
    if continuous_contact and len(contact) > 1:
        _cov = np.zeros(T, dtype=bool)
        for _c in contact.values():
            _cov |= _c
        if not _cov.all():
            _bas = {sd: np.nanmin(result[:, side_markers[sd], 1], axis=1)
                    for sd in contact}
            _min_gap = 3
            _i = 0
            while _i < T:
                if _cov[_i]:
                    _i += 1
                    continue
                _j = _i
                while _j < T and not _cov[_j]:
                    _j += 1
                if _j - _i >= _min_gap:
                    # Un seul proprietaire pour tout le trou — le pied le plus
                    # bas en moyenne, celui qui porte la charge. Choisir image
                    # par image ferait battre l'ancre d'un pied a l'autre.
                    _own = min(contact, key=lambda sd: float(
                        np.nanmean(_bas[sd][_i:_j])))
                    contact[_own][_i:_j] = True
                _i = _j

    # ── ANCRE A L'ATTERRISSAGE, PUIS POSE PROGRESSIVE ────────────────────
    # Portage de `M2S:contact_optim.py:refine_contacts`, deux idees distinctes.
    #
    # 1. L'ancre est la ou le pied ATTERRIT : la mediane des premieres images de
    #    l'appui, pas la position instantanee de la premiere image. Celle-ci
    #    porte tout le bruit de l'impact, et l'ancre en herite pour toute la
    #    phase.
    # 2. L'ANCRE N'EST PAS CONSTANTE : UN PIED QUI SE POSE N'EST PAS UN PIED
    #    ARRETE. Mesure de Mesh2Sim sur le gold BioCV P03, appuis definis par
    #    les PLATEFORMES donc sans seuil : apres le contact le talon avance
    #    ENCORE de 4,4 cm et l'orteil de 8,6 a 9,1 cm, progressivement sur 100 a
    #    150 ms, puis s'arrete. Une ancre figee des la premiere image ecrase ce
    #    deplacement reel et produit la pose saccadee. La cible glisse donc de
    #    la position d'entree vers la position stabilisee suivant un quart de
    #    sinusoide sur `settle_s` — forme verifiee chez eux : sin(pi/2 u) predit
    #    0,32/0,61/0,83/0,97 a 25/50/75/100 ms contre 0,23/0,61/0,80/0,91
    #    observes — puis tient. Le verrouillage de mi-appui, le vrai
    #    anti-patinage, est INCHANGE.
    #
    # La pose est FORCEMENT VERS L'AVANT et bornee a `max_settle_m` : sur des
    # donnees monoculaires bruitees le deplacement des 120 ms suivant le contact
    # peut sortir negatif, ce qui ferait RECULER un pied pose — physiquement
    # impossible, et pire que de le figer. Le lateral et le vertical sont
    # ecartes : Mesh2Sim a mesure deux fois que suivre le lateral degrade
    # l'adduction de hanche (r 0,93 -> 0,70 sur P06), la hanche etant le seul
    # degre de liberte capable d'absorber une contrainte laterale au pied.
    _phases: dict[str, list[tuple[int, int]]] = {}
    _anc_raw: dict[str, np.ndarray] = {}
    _poids: dict[str, np.ndarray] = {}
    _ns = max(1, int(round(settle_s * fps)))
    _nr = max(1, int(round(ramp_s * fps)))
    _rampe = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, _nr + 2)[1:-1]))
    for side in contact:
        _ph, _k0 = [], 0
        while _k0 < T:
            if contact[side][_k0]:
                _k1 = _k0
                while _k1 < T and contact[side][_k1]:
                    _k1 += 1
                _ph.append((_k0, _k1)); _k0 = _k1
            else:
                _k0 += 1
        _phases[side] = _ph
        _a = np.zeros((T, 2), dtype=np.float64)
        _w = np.zeros(T, dtype=np.float64)
        xz_s = foot_xz[side]
        for (a_, b_) in _ph:
            _kk = int(np.clip(int(0.15 * (b_ - a_)), 3, 10))
            _entree = np.nanmedian(xz_s[a_:min(b_, a_ + _kk)], axis=0)
            _a[a_:b_] = _entree
            if walk_axis is not None and not en_place and b_ - a_ > _ns + 2:
                _s1 = min(T, a_ + _ns + _kk)
                _pose = np.nanmedian(xz_s[max(a_ + _ns, _s1 - _kk):_s1], axis=0)
                _av = float(np.dot(_pose - _entree, walk_axis))
                _av = float(np.clip(_av, 0.0, max_settle_m * _unit))
                _u = np.clip((np.arange(a_, b_) - a_) / float(_ns), 0.0, 1.0)
                _a[a_:b_] = (_entree[None, :]
                             + np.sin(_u * (np.pi / 2))[:, None]
                             * (_av * walk_axis)[None, :])
            # Rampe en cosinus sureleve aux deux bouts : sans elle le serrage
            # s'etablit d'un coup et laisse un « tac » dans la trajectoire.
            _w[a_:b_] = 1.0
            _n = min(_nr, (b_ - a_) // 2)
            if _n > 0:
                _w[a_:a_ + _n] = _rampe[:_n]
                _w[b_ - _n:b_] = _rampe[:_n][::-1]
        _anc_raw[side] = _a
        _poids[side] = _w

    # ── VARIANCE D'APPUI, SANS ANCRE ─────────────────────────────────────
    # Formulation que Mesh2Sim teste actuellement (`contact_loss="variance"`,
    # venue d'OpenCap-Monocular), et dont leur note dit l'essentiel : « l'ancre
    # dit au pied OU se poser : elle est derivee de la pose d'entree, donc elle
    # porte deja notre erreur et tire le pied vers elle. La variance ne demande
    # que l'IMMOBILITE pendant l'appui — ce que le monoculaire observe bien — et
    # laisse le solveur choisir la position. »
    #
    # C'etait exactement notre defaut : mesure sur `outputs/GAIT_COULOIR`, la
    # derive de mi-appui du pied droit passait de 10,1 a 16,5 cm APRES
    # correction, l'ancre issue d'un atterrissage bruite tirant le pied vers un
    # mauvais endroit.
    #
    # Chez eux la variance est un terme de perte dans une optimisation
    # differentiable de la pose et de la racine. Chez nous la correction est un
    # DECALAGE GLOBAL RIGIDE s(t), donc le meme critere se resout en FORME
    # CLOSE, sans solveur ni gradient. On minimise
    #
    #     Somme_appuis (1/W) Somme_t w_t || x_t + s_t - moyenne_appui ||^2
    #   + lambda Somme_t || s_t - 2 s_{t-1} + s_{t-2} ||^2
    #   + mu     Somme_t || s_t ||^2
    #
    # Le premier terme est la variance ponderee de la position du pied pendant
    # chaque appui, normalisee par appui pour qu'un appui long ne pese pas plus
    # qu'un appui court — leur choix, repris tel quel. Le deuxieme est le terme
    # de continuite temporelle qu'ils decrivent comme « celui qui manquait » :
    # sans lui une pose sautillante passe telle quelle. Le troisieme retient la
    # correction pres de zero, pour ne pas reconstruire la trajectoire depuis
    # les pieds — ce que fait Mesh2Sim mais qui entrerait ici en conflit avec
    # `cam_t`.
    #
    # La moyenne d'appui s'elimine analytiquement, le probleme devient
    # quadratique en s et separable par axe :
    #     (A + lambda L^T L + mu I) s = -A x
    # avec A = Somme_appuis (1/W)(diag(w) - w w^T / W), semi-definie positive.
    if contact_loss == "variance":
        def _resoudre(axe: int) -> np.ndarray:
            A = np.zeros((T, T), dtype=np.float64)
            b = np.zeros(T, dtype=np.float64)
            for side in contact:
                x = foot_xz[side][:, axe]
                for (a_, b_) in _phases[side]:
                    w = _poids[side][a_:b_].copy()
                    ok = np.isfinite(x[a_:b_])
                    w[~ok] = 0.0
                    W = float(w.sum())
                    if W < 2.0:
                        continue
                    idx = np.arange(a_, b_)
                    A[np.ix_(idx, idx)] += (np.diag(w) - np.outer(w, w) / W) / W
                    xv = np.nan_to_num(x[a_:b_])
                    b[idx] += ((np.diag(w) - np.outer(w, w) / W) / W) @ xv
            if not np.any(A):
                return np.zeros(T)
            # Second difference (acceleration de la correction).
            L = (np.eye(T, k=0) - 2 * np.eye(T, k=1) + np.eye(T, k=2))[:max(T - 2, 0)]
            M = A + var_smooth * (L.T @ L) + var_prior * np.eye(T)
            try:
                return np.linalg.solve(M, -b)
            except np.linalg.LinAlgError:      # pragma: no cover
                return np.zeros(T)

        shifts = np.column_stack([_resoudre(0), _resoudre(1)])
        if max_shift_m > 0:
            _n = np.linalg.norm(shifts, axis=1)
            _tf = np.where(_n > max_shift_m * _unit,
                           (max_shift_m * _unit) / np.maximum(_n, 1e-9), 1.0)
            shifts = shifts * _tf[:, None]
        result[:, :, 0] += shifts[:, 0][:, None]
        result[:, :, 2] += shifts[:, 1][:, None]
        return result, shifts / _unit

    shifts = np.zeros((T, 2), dtype=np.float64)
    # Plafond exprimé en VITESSE (donc indépendant du fps). Volontairement
    # serré : notre déplacement global vient de `cam_t`, et une correction
    # ample reviendrait à reconstruire la trajectoire depuis les pieds —
    # ce que fait Mesh2Sim, mais qui entrerait ici en conflit avec cam_t
    # (mesuré : 3,95 m de correction sur un couloir de 10,8 m). On se limite
    # donc à retirer la gigue.
    max_step = max_contact_speed_ms * _unit / max(fps, 1e-6)
    anchors: dict[str, np.ndarray | None] = {s: None for s in contact}
    # Nombre d'images consecutives hors contact, par pied : l'ancre ne meurt
    # qu'apres `anchor_hold_s` (0 en locomotion, ou tout vol est reel).
    off: dict[str, int] = {s: 0 for s in contact}
    hold = int(round(anchor_hold_s * fps))

    # Decalage fige a l'entree de chaque phase : l'ancre pre-calculee est en
    # coordonnees BRUTES, on lui ajoute le decalage courant au moment de la pose
    # pour que chaque nouveau pas reparte du corps deja corrige.
    _off_phase: dict[str, np.ndarray] = {s: np.zeros(2) for s in contact}
    _phase_en_cours: dict[str, int] = {s: -1 for s in contact}
    for t in range(T):
        needed, poids = [], []
        for side in contact:
            xz = foot_xz[side][t]
            if not (contact[side][t] and np.all(np.isfinite(xz))):
                off[side] += 1
                if off[side] > hold:
                    anchors[side] = None  # pied en vol → l'ancre expire
                    _phase_en_cours[side] = -1
                continue
            off[side] = 0
            _pid = next((k for k, (a_, b_) in enumerate(_phases[side])
                         if a_ <= t < b_), -1)
            if _pid != _phase_en_cours[side]:
                _phase_en_cours[side] = _pid
                _off_phase[side] = (shifts[t - 1].copy() if t > 0
                                    else np.zeros(2))
            anchors[side] = _anc_raw[side][t] + _off_phase[side]
            _w = float(_poids[side][t])
            if _w <= 0.0:
                continue
            needed.append(anchors[side] - xz)
            poids.append(_w)

        if not needed:
            shifts[t] = shifts[t - 1] if t > 0 else 0.0
            continue

        # Double appui : moyenne PONDEREE par la rampe. Suivre un seul pied
        # ferait basculer le corps à chaque transition d'appui ; la rampe fait
        # passer le relais progressivement de l'un à l'autre.
        _pw = np.asarray(poids, dtype=np.float64)
        target = (np.stack(needed) * _pw[:, None]).sum(axis=0) / _pw.sum()
        step = target - (shifts[t - 1] if t > 0 else 0.0)
        # Un pied en appui ne glisse pas vite : au-delà, c'est de la gigue
        # d'inférence, et la compenser d'un coup projetterait le corps de côté.
        n = float(np.linalg.norm(step))
        # PLAFOND SELON LA DUREE DE L'APPUI, pas selon la vitesse du bassin.
        #
        # Le plafond de 0,30 m/s protege d'une correction ample qui
        # reconstruirait la trajectoire depuis les pieds et entrerait en
        # conflit avec `cam_t`. Mais sur un geste en place il EMPECHE de
        # planter le pied : mesure sur `outputs/SQUAT_VY` (2026-09-09), le
        # corps entier derive a 0,58 m/s pendant l'appui — dont 0,16 seulement
        # relatif au bassin, donc les trois quarts sont un mouvement commun,
        # precisement ce qu'un decalage global sait retirer — et la correction,
        # bridee a 0,30, ne rattrape jamais : 30, 47 puis 64 cm de derive nette
        # par phase d'appui. Florian : « faut bloquer des qu'y'a le contact au
        # sol, eviter que ca glisse ».
        #
        # Le discriminant n'est pas la vitesse du bassin — elle est contaminee
        # par le glissement lui-meme, le raisonnement tournerait en rond — mais
        # la DUREE de l'appui : une foulee de course dure 0,1 a 0,25 s, un pied
        # de squat, de lever de chaise ou de station debout reste pose des
        # secondes. Au-dela de `planted_min_s` le pied EST la reference et la
        # correction peut converger ; en deca, prudence historique. Course et
        # depart de sprint, dont les appuis sont brefs, ne changent pas d'un
        # millimetre.
        _plante = any(planted[side][t] and contact[side][t] for side in contact)
        _cap = planted_speed_ms * _unit / max(fps, 1e-6) if _plante else max_step
        if n > _cap:
            step *= _cap / n
        shifts[t] = (shifts[t - 1] if t > 0 else 0.0) + step

    # Lissage zéro-phase de la correction cumulée : les transitions d'appui
    # laissent des angles vifs dans la trajectoire du bassin. 2 Hz laisse passer
    # la cadence de marche (~1 Hz) tout en supprimant les ruptures.
    #
    # ⚠️ 2 Hz EST TROP BAS POUR UN GESTE EN PLACE, et c'etait la vraie cause du
    # glissement residuel du squat. Mesure sur `outputs/SQUAT_XZ` (2026-09-09) :
    # a t=1,6 s tout le corps se translate de 18 cm et revient en 0,24 s, soit
    # environ 4 Hz — la correlation entre la vitesse du pied et celle du bassin
    # vaut 0,97, et 0,95 avec C7, donc c'est bien un saut de la SCENE entiere,
    # exactement ce que le decalage global doit annuler. La consigne le
    # demandait correctement (12,5 cm) ; le filtre n'en laissait passer que 2,9.
    # En place il n'y a aucune cadence de marche a preserver : on monte a 8 Hz,
    # ce qui suit ces sauts tout en retirant le bruit image a image. La
    # locomotion garde 2 Hz, ou les transitions d'appui sont reelles.
    if T > 18 and fps > 6 and shift_lowpass_hz > 0:
        try:
            from scipy.signal import butter, sosfiltfilt
            wn = min(shift_lowpass_hz / (fps / 2.0), 0.99)
            shifts = sosfiltfilt(butter(2, wn, btype="low", output="sos"),
                                 shifts, axis=0)
        except Exception:
            pass  # scipy absent ou signal trop court : on garde le brut

    if max_shift_m > 0:
        _n = np.linalg.norm(shifts, axis=1)
        _tf = np.where(_n > max_shift_m * _unit,
                       (max_shift_m * _unit) / np.maximum(_n, 1e-9), 1.0)
        shifts = shifts * _tf[:, None]

    result[:, :, 0] += shifts[:, 0][:, None]
    result[:, :, 2] += shifts[:, 1][:, None]
    # shifts retournés en MÈTRES (pour appliquer le même décalage au mesh GLB,
    # qui est en mètres), quelle que soit l'unité de markers_array.
    return result, shifts / _unit

def lisser_trajectoire_rigide(
    markers_array: np.ndarray,
    marker_names: list,
    fps: float = 30.0,
    cutoff_hz: float = 2.5,
    max_shift_m: float = 0.60,
):
    """Retire les SAUTS de la trajectoire globale, par translation rigide.

    Constat qui a motive cet etage, mesure sur `outputs/GAIT_COULOIR` (marche en
    couloir, camera fixe, 2026-09-09) :

      * le bassin saute de **19 cm en une image** a t = 1,03 s, soit 11,2 m/s,
        et 41 images sur 344 depassent 4 m/s alors que la marche se fait a
        1,2 m/s ;
      * **aucun pied ne reste immobile plus de 0,28 s**, quand un appui de
        marche en dure 0,6 a 0,7.

    Autrement dit il n'y a, sur cette video, aucun appui identifiable a
    verrouiller : l'anti-glissement n'a rien de solide sur quoi s'appuyer, et
    aucun reglage de seuil ne le lui donnera. Le goulot est la trajectoire
    globale issue de `cam_t`, pas l'etage de contact.

    D'ou cet etage volontairement simple, et deliberement en amont : on lisse la
    trajectoire du bassin a ``cutoff_hz`` sans phase, et on applique l'ecart
    comme une TRANSLATION RIGIDE de tout le corps. Deux proprietes en decoulent,
    qui sont la raison de ce choix :

      * **aucun angle articulaire ne peut changer** — une translation rigide ne
        modifie aucune distance ni aucun angle interne. Le risque de degrader
        la biomecanique est nul par construction, ce qui n'est pas le cas d'un
        verrouillage de pied ;
      * la vitesse moyenne de progression est conservee (un filtre passe-bas
        garde la composante continue), donc la longueur de foulee et la vitesse
        restent celles qui sont mesurees.

    2,5 Hz laisse passer la cadence de pas (environ 2 Hz) et l'oscillation
    avant-arriere du bassin, tout en coupant les sauts. Mesure :

        essai            vitesse max du bassin      derive de mi-appui G / D
        GAIT_COULOIR     11,17 -> 6,92 m/s          11,1 -> 8,6  /  10,1 -> 10,0 cm
        REG_run          14,25 -> 8,38 m/s           9,4 -> 9,7  /  40,2 -> 21,6 cm
        SPRINT (valide)   6,67 -> 5,88 m/s           6,0 -> 5,0  /   1,7 ->  2,0 cm
        REG_hop           5,39 -> 3,66 m/s          14,5 -> 12,5 /  79,8 -> 74,7 cm

    Le deplacement net est conserve au centimetre sur les quatre.

    Returns:
        (markers_corriges, shifts) — shifts en METRES, (T, 2) en XZ, a rejouer
        tel quel sur le mesh comme les decalages de l'anti-glissement.
    """
    if markers_array is None or markers_array.ndim != 3 or markers_array.shape[0] < 20:
        return markers_array, None
    T = markers_array.shape[0]
    _unit = 1000.0 if np.nanmax(np.abs(markers_array)) > 50.0 else 1.0
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    pel = [name_to_idx[m] for m in ("RASI", "LASI", "RPSI", "LPSI")
           if m in name_to_idx]
    if not pel:
        return markers_array, None
    p = np.nanmean(markers_array[:, pel, :][:, :, [0, 2]], axis=1)  # (T, 2)
    ok = np.all(np.isfinite(p), axis=1)
    if ok.sum() < 10:
        return markers_array, None
    idx = np.arange(T, dtype=np.float64)
    p = np.column_stack([np.interp(idx, idx[ok], p[ok, j]) for j in range(2)])
    if not (0 < cutoff_hz < 0.5 * fps):
        return markers_array, None
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, cutoff_hz / (0.5 * fps), btype="low", output="sos")
        q = sosfiltfilt(sos, p, axis=0)
    except ImportError:                                    # pragma: no cover
        return markers_array, None
    shifts = q - p
    # Garde-fou : au-dela, ce n'est plus un saut a lisser mais un changement de
    # sujet ou une perte de suivi, et translater le corps d'autant serait pire
    # que de ne rien faire.
    n = np.linalg.norm(shifts, axis=1)
    lim = max_shift_m * _unit
    if np.any(n > lim):
        shifts = shifts * np.where(n > lim, lim / np.maximum(n, 1e-9), 1.0)[:, None]
    out = markers_array.copy()
    out[:, :, 0] += shifts[:, 0][:, None]
    out[:, :, 2] += shifts[:, 1][:, None]
    return out, shifts / _unit


def lateral_root_shift(
    markers_array: np.ndarray,
    marker_names: list,
    fps: float = 30.0,
    settle_s: float = 0.30,
    band_m: float = 0.030,
    min_contact_s: float = 0.06,
    max_contact_s: float = 1.00,
    max_corr_m: float = 0.30,
    min_travel_m: float = 0.50,
):
    """Recale LATÉRALEMENT le corps entier, à appliquer APRÈS l'IK.

    Mesuré le 2026-09-08 sur `Sprint_start` et `IMG_6224` : la dérive du pied
    pendant l'appui ne vit PAS le long de l'axe d'avance mais sur le côté —
    7,5 cm en latéral contre 4,2 cm en progression. C'est pourquoi l'« axe X
    seul » de Mesh2Sim ne se transpose pas tel quel : X est LEUR axe d'avance.
    Corriger la seule progression ne retire quasi rien (9,71 → 7,84 cm) ;
    corriger le seul latéral retire l'essentiel (→ 4,16 cm).

    On ne touche qu'à la composante latérale, délibérément : la progression est
    la mesure (vitesse, longueur de foulée) et la réécrire reviendrait à
    reconstruire la trajectoire depuis les pieds. Le mouvement de côté, lui,
    n'est mesuré par rien et n'est que du bruit de reconstruction.

    La transformation est une TRANSLATION RIGIDE : elle ne change aucun angle
    articulaire (vérifié, écart max 7e-14°). C'est ce qui la rend applicable
    APRÈS l'IK — il n'y a plus de moindres carrés derrière pour la défaire, et
    elle ne met aucun marqueur en concurrence avec les 40 autres. C'est la
    différence de fond avec une correction pré-IK, qui elle se fait laver.

    `settle_s` est la courbe de tassement (moyenne glissante centrée). Sans
    elle, l'épinglage est dur : dérive nulle, mais accélération du bassin de
    32 → 151 m/s², soit un pas de 5,7 m/s au pic — le corps est téléporté.
    À 0,30 s la dérive tombe à 4,05 cm POUR UN COÛT NUL (33 contre 32 m/s²).

    Deux garde-fous, tous deux motivés par un échec mesuré (`wk_squat`, sujet
    qui marche 1 m puis s'accroupit) : l'appui de 1,8 s du squat était pris pour
    un pas, la correction montait à 40 cm et la dérive EMPIRAIT (10,9 → 11,4 cm).
      • `max_contact_s` — au-delà d'une seconde, ce n'est pas un pas, c'est une
        station debout. Le « glissement » qu'on y mesure est l'oscillation
        posturale réelle du sujet, qu'il ne faut surtout pas annuler.
      • `max_corr_m` — filet de sécurité contre l'emballement, pas un réglage.
        Mesuré : c'est la DURÉE qui fait tout le travail (squat marché 11,4 →
        9,2 cm), l'amplitude seule ne corrige rien et, serrée à 15 cm, elle
        rejette de VRAIS appuis de sprint (5,86 → 7,65 cm). Laissée large.

    Returns:
        (markers_corrigés, shifts) — shifts (T, 2) = (dx, dz) en MÈTRES, à
        appliquer aussi à `pelvis_tx` / `pelvis_tz` du `.mot` pour garder les
        deux sorties cohérentes.
    """
    if markers_array is None or markers_array.ndim != 3 or markers_array.shape[0] < 4:
        return markers_array, None
    result = markers_array.copy()
    T = result.shape[0]
    unit = 1000.0 if np.nanmax(np.abs(markers_array)) > 50.0 else 1.0
    name_to_idx = {n: i for i, n in enumerate(marker_names)}

    # Axe de progression = déplacement net du bassin. Sans avancée franche, le
    # « latéral » n'a pas de définition : on ne corrige rien plutôt que de
    # corriger dans une direction arbitraire.
    pel = [name_to_idx[m] for m in ("RASI", "LASI", "RPSI", "LPSI")
           if m in name_to_idx]
    if not pel:
        return result, np.zeros((T, 2))
    pelvis_xz = np.nanmean(result[:, pel, :][:, :, [0, 2]], axis=1)
    net = np.nanmean(pelvis_xz[-5:], axis=0) - np.nanmean(pelvis_xz[:5], axis=0)
    if not np.all(np.isfinite(net)) or np.linalg.norm(net) < min_travel_m * unit:
        return result, np.zeros((T, 2))
    fwd = net / np.linalg.norm(net)
    lat = np.array([-fwd[1], fwd[0]])          # normale à l'axe d'avance, dans XZ

    # Points de contact : la SEMELLE. Les marqueurs cutanés ne descendent jamais
    # sous ~3 cm du sol ; seuls les points plantaires touchent réellement, et
    # c'est ce qui rend un seuil de hauteur utilisable.
    episodes = []
    for side in ("r", "l"):
        pts = [i for n, i in name_to_idx.items()
               if n.startswith("SOLE_") and n.endswith("_" + side)]
        if not pts:
            continue
        y = np.nanmin(result[:, pts, 1], axis=1)
        if not np.isfinite(y).any():
            continue
        on = (y - np.nanmin(y)) <= band_m * unit
        min_len = max(3, int(round(min_contact_s * fps)))
        max_len = max(min_len + 1, int(round(max_contact_s * fps)))
        i = 0
        while i < T:
            if on[i]:
                j = i
                while j < T and on[j]:
                    j += 1
                if min_len <= (j - i) <= max_len:
                    episodes.append((i, j, pts))
                i = j
            else:
                i += 1
    if not episodes:
        return result, np.zeros((T, 2))
    episodes.sort()

    # Ancre par épisode, en coordonnées DÉJÀ corrigées : le décalage acquis au
    # pas précédent est conservé, sinon chaque pose rappellerait le corps en
    # arrière et la marche n'avancerait plus.
    shifts = np.zeros((T, 2), dtype=np.float64)
    carry = np.zeros(2)
    for a, b, pts in episodes:
        seg = result[a:b][:, pts, :]
        # Référence = le point plantaire le plus bas de TOUT l'épisode, fixe.
        # Suivre « le plus bas par image » ferait sauter la référence d'un point
        # à l'autre à la bascule talon→orteil.
        low = int(np.nanargmin(np.nanmin(seg[:, :, 1], axis=0)))
        h = seg[:, low, :][:, [0, 2]]
        ref = h[0]
        corr = np.zeros((b - a, 2))
        for k in range(b - a):
            if np.all(np.isfinite(h[k])):
                corr[k] = ((ref - h[k]) @ lat) * lat      # latéral SEULEMENT
        if float(np.abs(corr).max()) > max_corr_m * unit:
            continue                                     # pas un glissement
        shifts[a:b] += carry + corr
        carry = carry + corr[-1]
        shifts[b:] += corr[-1]

    # Courbe de tassement : moyenne glissante centrée, zéro-phase.
    win = max(1, int(round(settle_s * fps)))
    if win > 1:
        for j in range(2):
            shifts[:, j] = uniform_filter1d(shifts[:, j], size=win, mode="nearest")

    result[:, :, 0] += shifts[:, 0][:, None]
    result[:, :, 2] += shifts[:, 1][:, None]
    return result, shifts / unit
