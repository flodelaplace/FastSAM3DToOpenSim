"""Sol stable : plan du sol ajuste une fois pour l'essai, rotation GLOBALE de la
scene, offset constant. Portage de l'etape 3 de ``ground_anchor`` de Mesh2Sim
(`stages/ground_anchor/src/mesh2sim_ground_anchor/mono.py:836-846`), SANS son
etape 5 (correction verticale par image) qui suppose qu'un pied est toujours au
sol et colle donc le sujet au sol pendant une phase de vol.

POURQUOI CE MODULE EXISTE
-------------------------
Le pipeline redressait la scene en faisant tourner le SQUELETTE autour du BASSIN
de chaque image (``_rotate_around_pelvis_z`` / ``_x``), d'une FRACTION du pitch
camera mesure (``LEAN_SCALE = 0.5``). Une rotation autour du bassin de l'image t
s'ecrit ``x -> R(x - p_t) + p_t = R x + (I - R) p_t`` : le terme ``(I - R) p_t``
suit le bassin et annule exactement l'effet de la rotation sur la TRAJECTOIRE.
Elle redresse donc la pose et laisse le sol incline. Sur un sujet qui parcourt D
metres avec un sol vu incline de theta, la hauteur du sol derive de D.sin(theta)
— 2,6 cm par metre a 1,5 deg. C'est ce qui rend toute detection de contact par
hauteur impossible chez nous, et c'est la seule raison pour laquelle le detecteur
de course utilise Zeni (position relative au sacrum), donc depend du bassin, donc
change de reponse selon que la translation globale est figee ou non.

Ici la rotation est GLOBALE (un pivot fixe pour tout l'essai) et le decalage
vertical est UNE CONSTANTE. Les deux operations forment une transformation rigide
unique appliquee identiquement a toutes les images : les angles articulaires
internes (genou, cheville, hanche) en sont donc RIGOUREUSEMENT invariants. Seules
changent les grandeurs referencees au monde (pelvis_tilt/list/rotation, hauteurs).

CONVENTION DE SIGNE — une seule dans tout ce fichier
----------------------------------------------------
Repere monde du pipeline : X avant, Y haut, Z lateral (droite), metres.
Les matrices de rotation sont en FORME COLONNE (``R @ v_colonne``) et s'appliquent
a des tableaux de points en lignes par ``pts @ R.T``. Jamais ``pts @ R``.
Le fichier ``coordinate_transform.py`` melange les deux conventions
(``_rotate_around_pelvis_z`` fait ``v @ Rz``, donc -theta ; ``_apply_body_vertical_correction``
fait ``v @ R.T``, donc +theta) ; ne pas importer cette ambiguite ici.
Une normale est un vecteur LIBRE EN SIGNE : la nier pour la remettre vers le haut
est legitime. Nier un REPERE entier serait une reflexion (det = -1) et inverserait
le roulis — le piege qui a coute 22 deg a Mesh2Sim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "GroundPlaneFit",
    "rotation_align_a_to_b",
    "fit_floor_plane",
    "StableFloor",
    "stable_floor_transform",
    "apply_stable_floor",
    "detect_contact_by_height",
    "contact_runs_to_mask",
]


# ---------------------------------------------------------------------------
# geometrie
# ---------------------------------------------------------------------------

def rotation_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation (forme colonne) la plus courte amenant le vecteur ``a`` sur ``b``.

    Rodrigues. Renvoie l'identite si les deux sont deja colineaires, et une
    rotation de pi autour d'un axe orthogonal arbitraire s'ils sont opposes.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # antipodal : demi-tour autour de n'importe quel axe orthogonal a `a`
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * K @ K
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


@dataclass
class GroundPlaneFit:
    """Resultat de l'ajustement du plan du sol sur les points plantaires bas."""
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt_deg: float = 0.0
    # pentes exprimees dans le repere horizontal de l'essai, pas dans X/Z monde
    slope_travel_deg: float = 0.0   # pente le long de l'axe de progression
    slope_cross_deg: float = 0.0    # pente perpendiculaire (rarement observable)
    travel_axis_xz: list = field(default_factory=lambda: [1.0, 0.0])
    n_candidates: int = 0
    n_inliers: int = 0
    span_travel_m: float = 0.0
    span_cross_m: float = 0.0
    span_x_m: float = 0.0
    span_z_m: float = 0.0
    pitch_trusted: bool = False     # pente de progression observee ?
    roll_trusted: bool = False      # pente transverse observee ?
    ok: bool = False
    reason: str = ""


def _candidate_floor_points(
    plantar: np.ndarray,
    per_foot_split: int | None = None,
) -> np.ndarray:
    """Points candidats sol = le point plantaire LE PLUS BAS de chaque pied a
    chaque image.

    ``plantar`` : (T, P, 3). ``per_foot_split`` : nombre de points du premier
    pied (les suivants sont l'autre pied) ; ``None`` = un seul groupe.

    On ne prend pas les P points de chaque image : en vol ils sont tous en l'air,
    en appui ils sont quasi coplanaires, donc le point le plus bas par pied
    porte toute l'information utile et divise la taille du probleme par ~7.
    RANSAC se charge d'ecarter les images de vol (qui sont TOUTES au-dessus du
    plan, jamais en dessous).
    """
    groups: list[tuple[int, int]] = []
    P = plantar.shape[1]
    if per_foot_split and 0 < per_foot_split < P:
        groups = [(0, per_foot_split), (per_foot_split, P)]
    else:
        groups = [(0, P)]
    out = []
    for a, b in groups:
        sub = plantar[:, a:b, :]                       # (T, p, 3)
        y = sub[:, :, 1]
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(y).any(axis=1)
        if not ok.any():
            continue
        j = np.nanargmin(np.where(np.isfinite(y), y, np.inf), axis=1)
        pts = sub[np.arange(sub.shape[0]), j, :]
        out.append(pts[ok & np.isfinite(pts).all(axis=1)])
    if not out:
        return np.empty((0, 3))
    return np.concatenate(out, axis=0)


def fit_floor_plane(
    plantar: np.ndarray,
    *,
    per_foot_split: int | None = None,
    inlier_thresh_m: float = 0.025,
    n_iters: int = 3000,
    max_tilt_deg: float = 20.0,
    min_span_travel_m: float = 0.80,
    min_span_cross_m: float = 1.50,
    min_inliers: int = 12,
    seed: int = 0,
) -> GroundPlaneFit:
    """Ajuste la hauteur du sol par RANSAC sur les points plantaires bas.

    L'ajustement se fait dans un repere horizontal PROPRE A L'ESSAI et non dans
    X/Z du monde : on prend l'axe principal de l'etalement horizontal des points
    de contact (``u``, l'axe de progression) et sa perpendiculaire (``w``). On
    ajuste ``y = a.s + b.q + c`` avec ``s`` et ``q`` les coordonnees le long de
    ``u`` et ``w``.

    POURQUOI CE DECOUPAGE, ET POURQUOI LES DEUX SEUILS SONT DIFFERENTS
    ------------------------------------------------------------------
    Le long de l'axe de progression, le sujet balaie plusieurs metres : une
    derive de hauteur y est un VRAI defaut de verticale, et c'est exactement le
    ``D.sin(theta)`` qu'on cherche a annuler.

    Perpendiculairement, l'etendue des points de contact n'est PAS une etendue
    de sol : c'est la largeur du pas, ~10 a 20 cm de sol utile noyes dans
    l'ecart entre pied gauche et pied droit. Mesure sur `outputs/RUN_6224` :
    l'ajustement libre sur X/Z sortait un roulis de **+7,5 deg** pour 1,22 m
    d'etendue transverse, soit 16 cm de denivele entre les deux pieds. Aucun sol
    n'est incline de 7,5 deg : ce chiffre mesure une asymetrie gauche/droite de
    la reconstruction, et le « corriger » fait basculer le sujet sur le cote.
    Les deux quantites ne sont pas separables a partir des seuls points
    plantaires — c'est pour cela que Mesh2Sim tire sa verticale de l'IMAGE
    (GeoCalib / MoGe) et non des pieds, et que son mode `ransac_ground_plane`
    n'est pas le defaut.

    D'ou ``min_span_cross_m`` nettement plus exigeant que ``min_span_travel_m``.
    En pratique, sur une course en ligne, le roulis reste a zero et seul le
    pitch de progression est corrige. Le roulis reste le domaine de MoGe.

    Quand aucun des deux axes n'est observe (sujet en place : squat, STS, velo,
    ou traitement `--stationary`), l'ajustement ECHOUE explicitement plutot que
    d'inventer une inclinaison. L'appelant garde alors son comportement actuel.
    """
    res = GroundPlaneFit()
    pts = _candidate_floor_points(plantar, per_foot_split=per_foot_split)
    res.n_candidates = int(pts.shape[0])
    if pts.shape[0] < max(min_inliers, 8):
        res.reason = f"seulement {pts.shape[0]} points plantaires exploitables"
        return res

    # --- repere horizontal de l'essai : u = axe de progression, w = transverse
    hz = pts[:, [0, 2]]
    hz0 = hz - hz.mean(axis=0)
    _w, V = np.linalg.eigh(hz0.T @ hz0)
    u = V[:, -1] / (np.linalg.norm(V[:, -1]) + 1e-12)
    wv = np.array([-u[1], u[0]])
    s = hz0 @ u
    q = hz0 @ wv
    y = pts[:, 1]
    res.travel_axis_xz = u.tolist()
    res.span_travel_m = float(np.ptp(s))
    res.span_cross_m = float(np.ptp(q))
    res.span_x_m = float(np.ptp(pts[:, 0]))
    res.span_z_m = float(np.ptp(pts[:, 2]))
    res.pitch_trusted = res.span_travel_m >= min_span_travel_m
    res.roll_trusted = res.span_cross_m >= min_span_cross_m
    if not (res.pitch_trusted or res.roll_trusted):
        res.reason = (f"aucun axe observe : etendue progression "
                      f"{res.span_travel_m:.2f} m < {min_span_travel_m} m, "
                      f"transverse {res.span_cross_m:.2f} m < {min_span_cross_m} m")
        return res

    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    tan_max = np.tan(np.radians(max_tilt_deg))
    best_inl, best_cnt = None, 0
    for _ in range(n_iters):
        idx = rng.choice(n, 3, replace=False)
        A = np.column_stack([s[idx], q[idx], np.ones(3)])
        try:
            coef = np.linalg.solve(A, y[idx])
        except np.linalg.LinAlgError:
            continue
        a, b, c0 = coef
        if abs(a) > tan_max or abs(b) > tan_max:
            continue
        inl = np.abs(y - (a * s + b * q + c0)) < inlier_thresh_m
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None or best_cnt < min_inliers:
        res.reason = f"RANSAC : {best_cnt} inliers < {min_inliers}"
        return res

    # raffinement moindres carres sur les inliers, axe par axe selon ce qui est
    # reellement observe (un axe non observe n'est pas ajuste, il est mis a zero)
    cols, keys = [], []
    if res.pitch_trusted:
        cols.append(s[best_inl]); keys.append("a")
    if res.roll_trusted:
        cols.append(q[best_inl]); keys.append("b")
    cols.append(np.ones(best_cnt))
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), y[best_inl], rcond=None)
    vals = dict(zip(keys, coef))
    a = float(vals.get("a", 0.0))
    b = float(vals.get("b", 0.0))
    if abs(a) > tan_max or abs(b) > tan_max:
        res.reason = (f"inclinaison ajustee hors bornes (a={a:.3f}, b={b:.3f}, "
                      f"max tan={tan_max:.3f})")
        return res

    # normale : dans le repere (u, +Y, w), le plan est y = a.s + b.q + c donc la
    # normale vaut (-a, 1, -b) exprimee sur (u, Y, w). Retour en X/Y/Z.
    nu, nw = -a, -b
    normal = np.array([nu * u[0] + nw * wv[0], 1.0, nu * u[1] + nw * wv[1]])
    normal /= np.linalg.norm(normal)
    res.normal = normal
    res.n_inliers = best_cnt
    res.tilt_deg = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    res.slope_travel_deg = float(np.degrees(np.arctan(a)))
    res.slope_cross_deg = float(np.degrees(np.arctan(b)))
    res.pivot = pts[best_inl].mean(axis=0)
    res.ok = True
    return res


@dataclass
class StableFloor:
    """Transformation « sol stable » d'un essai : rotation globale + derive
    verticale + offset constant. Trois blocs, chacun avec un nombre FIXE de
    parametres, jamais un parametre par image."""
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drift_m_per_s: float = 0.0
    t0_frame: float = 0.0
    offset_m: float = 0.0
    fit: GroundPlaneFit = field(default_factory=GroundPlaneFit)
    n_poses: int = 0
    pose_floor_std_cm: float = float("nan")
    rotation_applied: bool = False
    drift_applied: bool = False
    drift_model: str = "linear"
    stance_shift_m: np.ndarray | None = None   # (T,) mode "stance" uniquement
    stance_coverage: float = 0.0               # part des images en appui detecte
    notes: str = ""


def _pose_floor_levels(plantar_y_by_foot: list[np.ndarray], fps: float,
                       min_cycle_s: float = 0.35, prominence_m: float = 0.02):
    """Hauteur du point plantaire le plus bas a chaque POSE, et son image.

    Une pose = un minimum local de la hauteur du point le plus bas d'un pied.
    Definition volontairement independante de tout detecteur de contact : c'est
    precisement ce qu'on cherche a rendre possible.
    """
    try:
        from scipy.signal import butter, filtfilt, find_peaks
    except ImportError:                                    # pragma: no cover
        return np.empty(0), np.empty(0)
    ts, ys = [], []
    for y in plantar_y_by_foot:
        if not np.isfinite(y).any():
            continue
        yy = np.nan_to_num(y, nan=float(np.nanmax(y)))
        # lissage 8 Hz avant la recherche de minima : sans lui, le bruit image a
        # image cree de faux minima locaux et la dispersion mesuree est celle du
        # bruit, pas celle du sol.
        if len(yy) > 20 and 8.0 < 0.5 * fps:
            bb, aa = butter(4, 8.0 / (0.5 * fps), btype="low")
            yy = filtfilt(bb, aa, yy)
        idx, _ = find_peaks(-yy, distance=max(2, int(min_cycle_s * fps)),
                            prominence=prominence_m)
        for k in idx:
            if np.isfinite(y[k]):
                ts.append(float(k)); ys.append(float(y[k]))
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(ys)[o]


def _robust_line(t: np.ndarray, y: np.ndarray, n_iter: int = 2):
    """Droite y = a.t + b, avec deux passes de rejet a 2 ecarts-types."""
    keep = np.ones(len(t), dtype=bool)
    a = b = 0.0
    for _ in range(n_iter + 1):
        if keep.sum() < 3:
            break
        A = np.column_stack([t[keep], np.ones(int(keep.sum()))])
        c, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        a, b = float(c[0]), float(c[1])
        r = y - (a * t + b)
        sd = float(np.std(r[keep]))
        if sd < 1e-9:
            break
        keep = np.abs(r) < 2.0 * sd
    return a, b, keep


def _stance_gated_shift(
    plantar: np.ndarray,
    split: int,
    fps: float,
    *,
    thresh_m: float = 0.020,
    min_run_s: float = 0.06,
    lowpass_hz: float = 4.0,
):
    """Correction verticale CONDITIONNEE A L'APPUI, interpolee pendant le vol.

    C'est la forme que le document de Mesh2Sim prescrit explicitement pour la
    course (§4, l. 98-100) : « garder l'etape 3 (sol constant sur l'essai) et
    supprimer ou CONDITIONNER l'etape 5 — ne l'appliquer qu'aux images ou un pied
    est detecte en appui, ou pas du tout ».

    Difference avec le clamp actuel du pipeline (`coordinate_transform.py:264-296`),
    qui est la version a ne pas faire :
      * ici la reference est la SEMELLE (points plantaires), pas la peau. Les
        marqueurs cutanes sont 3 a 4 cm au-dessus de la semelle et cet ecart
        VARIE pendant le deroule du pied : les prendre pour reference injecte une
        oscillation verticale parasite dans la reference elle-meme ;
      * la correction est BIDIRECTIONNELLE. Le clamp actuel ne remonte que, donc
        le bruit s'accumule vers le haut sur toute la duree de l'essai ;
      * les images de vol ne sont PAS corrigees, elles sont INTERPOLEES entre les
        deux appuis qui les encadrent. Un pied en l'air ne definit aucun sol ;
      * le resultat est lisse a ``lowpass_hz`` avant d'etre soustrait, pour ne pas
        transferer le bruit image a image de la detection dans le squelette.

    Ce que ce mode NE FAIT PAS : aplatir l'oscillation verticale du centre de
    masse. Pendant l'appui c'est le PIED qui est tenu au sol ; le bassin reste
    libre de monter et descendre au-dessus. Pendant le vol rien n'est tenu.

    Returns:
        (shift (T,), couverture d'appui) ou (None, couverture) si aucun appui.
    """
    T, P = plantar.shape[0], plantar.shape[1]
    groups = [(0, split), (split, P)] if 0 < split < P else [(0, P)]
    level = np.full(T, np.nan)
    covered = np.zeros(T, dtype=bool)
    for a, b in groups:
        for k in range(a, b):
            y = plantar[:, k, 1]
            if not np.isfinite(y).any():
                continue
            fl = float(np.nanpercentile(y, 1.0))
            for r0, r1 in detect_contact_by_height(
                    y, fl, thresh_m=thresh_m, min_run_s=min_run_s, fps=fps):
                seg = y[r0:r1]
                cur = level[r0:r1]
                level[r0:r1] = np.where(np.isnan(cur), seg, np.minimum(cur, seg))
                covered[r0:r1] = True
    cov = float(covered.mean())
    if not covered.any():
        return None, cov
    idx = np.arange(T, dtype=np.float64)
    good = np.isfinite(level)
    shift = np.interp(idx, idx[good], level[good])
    if lowpass_hz > 0 and T > 20 and lowpass_hz < 0.5 * fps:
        try:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(2, lowpass_hz / (0.5 * fps), btype="low", output="sos")
            shift = sosfiltfilt(sos, shift)
        except ImportError:                                # pragma: no cover
            pass
    return np.asarray(shift, dtype=np.float64), cov


def stable_floor_transform(
    plantar: np.ndarray,
    fps: float,
    *,
    per_foot_split: int | None = None,
    floor_percentile: float = 0.5,
    min_poses_for_drift: int = 4,
    min_drift_cm_per_s: float = 0.3,
    max_drift_cm_per_s: float = 15.0,
    correct_drift: bool = True,
    drift_model: str = "linear",
    stance_thresh_m: float = 0.020,
    stance_min_run_s: float = 0.06,
    stance_lowpass_hz: float = 4.0,
    stance_passes: int = 3,
    stance_max_shift_m: float = 0.30,
    stance_min_gain: float = 0.25,
    stance_min_coverage: float = 0.10,
    **fit_kwargs,
) -> StableFloor:
    """Calcule la transformation « sol stable » d'un essai.

    Trois blocs, appliques dans cet ordre :

    1. **Rotation globale** amenant la normale du sol ajuste sur +Y, autour d'un
       pivot FIXE. 3 parametres. C'est l'etape 3 de Mesh2Sim
       (`M2S:mono.py:836-846`). N'est appliquee que si la pente de progression
       est reellement observee (voir ``fit_floor_plane``).

    2. **Derive verticale de l'essai**, 1 parametre (cm/s), ajustee ROBUSTEMENT
       sur les seules hauteurs de POSE.

       Pourquoi ce bloc existe et n'a pas d'equivalent chez Mesh2Sim : notre
       mode ``--stationary`` injecte ``cam_t.Y`` dans le squelette
       (`coordinate_transform.py:158-167`) pour recuperer la descente du bassin
       en squat / STS. Filme a la main, ``cam_t.Y`` contient aussi le mouvement
       PROPRE de la camera. Mesure sur `outputs/SMOKE_FINAL` (IMG_6238, course,
       stationnaire) : le sol monte de **+4,4 cm/s** de facon monotone
       (r = +0,86), HTOP monte de la meme quantite (ecart-type 7,5 cm, r = +0,73)
       — c'est la camera qui monte, pas le sujet. Sur `RUN_6224_stationary` :
       +1,45 cm/s. Sur `RUN_6224` traite EN DEPLACEMENT, ou ``cam_t.Y`` n'est pas
       injecte : **-0,04 cm/s**, HTOP ecart-type 0,5 cm. Le mecanisme est donc
       identifie sans ambiguite.

       Ce bloc n'est PAS l'etape 5 de Mesh2Sim (correction par image), qui a T
       parametres et colle le sujet au sol pendant le vol. Une droite ne peut pas
       absorber une oscillation de 0,2 s : la phase de vol et le rebond du centre
       de masse survivent par construction. C'est la difference entre corriger
       une derive et effacer un signal.

    3. **Offset constant** mettant le sol a Y = 0. 1 parametre. Etape 3 de
       Mesh2Sim, telle quelle.

    Chaque image recoit ainsi une transformation RIGIDE : les angles articulaires
    internes sont invariants, seul le referencement au monde change.

    Args:
        plantar : (T, P, 3) points PLANTAIRES en metres, repere monde Y-haut.
        fps : cadence, necessaire pour exprimer la derive en cm/s.
    """
    sf = StableFloor()
    T = plantar.shape[0]
    split = per_foot_split if per_foot_split else plantar.shape[1]

    # --- 1. rotation
    fit = fit_floor_plane(plantar, per_foot_split=per_foot_split, **fit_kwargs)
    sf.fit = fit
    work = plantar
    if fit.ok and fit.tilt_deg > 0.1:
        sf.R = rotation_align_a_to_b(fit.normal, np.array([0.0, 1.0, 0.0]))
        sf.pivot = fit.pivot
        sf.rotation_applied = True
        work = (plantar.reshape(-1, 3) - sf.pivot) @ sf.R.T + sf.pivot
        work = work.reshape(plantar.shape)

    # --- 2. derive verticale, ajustee sur les poses
    feet = [np.nanmin(work[:, :split, 1], axis=1)]
    if split < work.shape[1]:
        feet.append(np.nanmin(work[:, split:, 1], axis=1))
    tpose, ypose = _pose_floor_levels(feet, fps)
    sf.n_poses = int(len(tpose))
    sf.drift_model = drift_model
    if len(ypose) > 1:
        sf.pose_floor_std_cm = float(np.std(ypose) * 100)

    # 2a. derive lineaire (1 parametre). Elle passe TOUJOURS avant le mode
    # "stance", parce que la detection de contact par hauteur a besoin d'un sol
    # deja debarrasse de sa derive monotone pour fonctionner : mesure sur
    # `outputs/SMOKE_FINAL`, la couverture d'appui tombe a 8 % des images tant
    # que les 17 cm de derive sont la, contre 46 % sur un essai deja stable.
    # C'est une dependance circulaire, et l'ordre la casse.
    if correct_drift and "linear" in drift_model:
        if len(tpose) >= min_poses_for_drift:
            a_per_frame, _b, _keep = _robust_line(tpose, ypose)
            rate_cm_s = a_per_frame * fps * 100.0
            if min_drift_cm_per_s <= abs(rate_cm_s) <= max_drift_cm_per_s:
                sf.drift_m_per_s = float(a_per_frame * fps)
                sf.t0_frame = float(np.mean(tpose))
                sf.drift_applied = True
            elif abs(rate_cm_s) > max_drift_cm_per_s:
                sf.notes += f"derive {rate_cm_s:+.1f} cm/s hors bornes, non corrigee ; "
        else:
            sf.notes += (f"seulement {len(tpose)} poses (< {min_poses_for_drift}), "
                         f"derive lineaire non estimee ; ")
    if sf.drift_m_per_s:
        tt = np.arange(T, dtype=np.float64)
        work = work.copy()
        work[:, :, 1] -= (sf.drift_m_per_s / fps * (tt - sf.t0_frame))[:, None]

    # 2b. correction conditionnee a l'appui, sur le resultat de 2a
    if correct_drift and "stance" in drift_model:
        # Plusieurs passes : la detection de contact et la correction du sol se
        # conditionnent l'une l'autre. Mesure sur `outputs/SMOKE_FINAL`, la
        # couverture d'appui passe de 8 % a la premiere passe a ~25 % a la
        # troisieme, et l'etendue du sol de 7,3 a 4,4 cm. La suite converge : les
        # passes suivantes ne bougent plus la correction de plus de 1 mm.
        total = np.zeros(T, dtype=np.float64)
        probe = work.copy()
        _f0 = [np.nanmin(work[:, :split, 1], axis=1)]
        if split < work.shape[1]:
            _f0.append(np.nanmin(work[:, split:, 1], axis=1))
        _t0v, y0 = _pose_floor_levels(_f0, fps)
        for _ in range(max(1, stance_passes)):
            shift, cov = _stance_gated_shift(
                probe, split, fps, thresh_m=stance_thresh_m,
                min_run_s=stance_min_run_s, lowpass_hz=stance_lowpass_hz)
            sf.stance_coverage = cov
            if shift is None:
                break
            total += shift
            probe = probe.copy()
            probe[:, :, 1] -= shift[:, None]
            if float(np.max(np.abs(shift))) < 0.001:
                break
        # GARDE-FOUS. Le mode "stance" postule que la semelle touche VRAIMENT le
        # sol. Sur un geste ou c'est faux il produit une correction absurde et
        # DEGRADE tout : mesure sur `outputs/SMOKE_birddog` (quadrupedie, pieds
        # jamais poses) la correction atteint 83,7 cm et l'oscillation verticale
        # du bassin passe de 4,4 a 16,5 cm ; sur `outputs/RUN_CYCLING` (pieds sur
        # les pedales) 16,8 cm. Deux bornes, donc : l'amplitude de la correction,
        # et la couverture d'appui. Hors de ces bornes on ne corrige PAS et on le
        # dit — un sol faux annonce est pire qu'un sol non corrige.
        amp = float(np.max(np.abs(total))) if np.any(total) else 0.0
        # VALIDATION SUR LA MESURE ELLE-MEME. L'amplitude ne suffit pas a separer
        # un cas legitime d'un cas absurde : `outputs/SMOKE_FINAL` (course main
        # levee, correction legitime) demande 23,1 cm quand `outputs/RUN_CYCLING`
        # (pieds sur les pedales, correction absurde) en demande 16,8. On tranche
        # donc sur le RESULTAT : la correction doit reellement stabiliser le sol.
        feet2 = [np.nanmin(probe[:, :split, 1], axis=1)]
        if split < probe.shape[1]:
            feet2.append(np.nanmin(probe[:, split:, 1], axis=1))
        _t2, y2 = _pose_floor_levels(feet2, fps)
        std0 = float(np.std(y0)) if len(y0) > 1 else float("nan")
        std1 = float(np.std(y2)) if len(y2) > 1 else float("nan")
        gain = (1.0 - std1 / std0) if (std0 and np.isfinite(std0)
                                       and np.isfinite(std1)) else float("nan")
        if not np.any(total):
            sf.notes += "mode stance : aucun appui detecte, non applique ; "
        elif amp > stance_max_shift_m:
            sf.notes += (f"mode stance REFUSE : correction de {amp*100:.1f} cm > "
                         f"{stance_max_shift_m*100:.0f} cm, l'hypothese « la semelle "
                         f"touche le sol » ne tient pas sur ce geste ; ")
        elif sf.stance_coverage < stance_min_coverage:
            sf.notes += (f"mode stance REFUSE : appui detecte sur seulement "
                         f"{sf.stance_coverage*100:.0f} % des images ; ")
        elif not (np.isfinite(gain) and gain >= stance_min_gain):
            sf.notes += (f"mode stance REFUSE : ne stabilise pas le sol "
                         f"(dispersion des poses {std0*100:.2f} -> {std1*100:.2f} cm, "
                         f"gain {0.0 if not np.isfinite(gain) else gain*100:.0f} % < "
                         f"{stance_min_gain*100:.0f} %) ; ")
        else:
            sf.stance_shift_m = total
            sf.drift_applied = True
            work = probe

    # --- 3. offset constant
    y = work[:, :, 1].reshape(-1)
    y = y[np.isfinite(y)]
    sf.offset_m = float(np.percentile(y, floor_percentile)) if y.size else 0.0
    return sf


def apply_stable_floor(pts: np.ndarray, sf: StableFloor, fps: float) -> np.ndarray:
    """Applique la transformation a un tableau (T, N, 3) en metres.

    Chaque image subit une transformation rigide (meme rotation pour toutes, plus
    une translation verticale qui varie lentement) : aucun angle articulaire
    interne n'en est modifie.
    """
    out = np.asarray(pts, dtype=np.float64).copy()
    T = out.shape[0]
    if sf.rotation_applied:
        out = ((out.reshape(-1, 3) - sf.pivot) @ sf.R.T + sf.pivot).reshape(out.shape)
    if sf.drift_m_per_s:
        tt = np.arange(T, dtype=np.float64)
        out[:, :, 1] -= (sf.drift_m_per_s / fps * (tt - sf.t0_frame))[:, None]
    if sf.stance_shift_m is not None:
        out[:, :, 1] -= sf.stance_shift_m[:T, None]
    out[:, :, 1] -= sf.offset_m
    return out


# ---------------------------------------------------------------------------
# contact par hauteur plantaire
# ---------------------------------------------------------------------------

def detect_contact_by_height(
    y: np.ndarray,
    floor_y: float,
    *,
    thresh_m: float = 0.020,
    min_run_s: float = 0.06,
    fps: float = 60.0,
) -> list[tuple[int, int]]:
    """Contact = ce point est a moins de ``thresh_m`` de SON PROPRE plancher.

    Portage direct de `M2S:contact_optim.py:101-141`. Physique, sans centile de
    hauteur ni porte de vitesse.

    Le seuil de 2 cm vient de leur mesure sur plateformes de force : au moment ou
    la plateforme se charge, le talon est deja 0,7 a 1,4 cm au-dessus de son
    plancher ; une bande de 1,2 cm rate donc le debut de l'appui.

    Le plancher est PAR POINT, pas global. Un point de la voute plantaire ne
    descend jamais aussi bas que le talon : mesure contre un plancher global il
    ne serait JAMAIS vu en contact (chez eux 2 a 6 points sur 14 selon le sujet).
    """
    y = np.asarray(y, dtype=np.float64) - float(floor_y)
    on = np.isfinite(y) & (y <= thresh_m)
    T = len(y)
    min_run = max(2, int(round(min_run_s * fps)))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < T:
        if on[i]:
            j = i
            while j < T and on[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def contact_runs_to_mask(runs: list[tuple[int, int]], n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for a, b in runs:
        m[a:min(b, n_frames)] = True
    return m


# NOTE — ``ensure_continuous_coverage`` (`M2S:contact_optim.py:145-206`) N'EST PAS
# PORTEE, et ne doit pas l'etre. Elle pose « en marche il y a toujours un pied au
# sol, donc un trou de detection est un rate » et attribue chaque image non
# couverte au point le plus bas. Chez eux elle est meme FORCEE des que SOLE est
# actif (`M2S:contact_optim.py:319`). En course, en saut et en hop unipodal la
# phase de vol est reelle : la couverture continue transformerait chaque vol en
# appui. Leur propre document le dit (§7, l. 169-171).
