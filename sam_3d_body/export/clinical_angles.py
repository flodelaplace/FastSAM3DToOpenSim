"""Compute clinically-relevant derived angles from OpenSim body world transforms.

Bridge entre l'IK natif (qui produit `walker_knee` 1-DOF + autres joints simples)
et la kiné clinique (qui veut voir des angles type valgus/varus du genou, rotation
cheville, foot progression, lean du tronc). Ces angles ne sont PAS des DOF
articulaires natifs (impossibles à mesurer fiablement avec markers de surface)
mais des **angles segmentaires géométriques** calculés à partir des positions et
orientations 3D des bodies (sortie de l'IK).

Convention d'angles dérivés (signes choisis selon usage clinique courant) :

    knee_valgus_r/l         : frontal plane femur-tibia.
                              Positive = valgus (knee inward / vers le centre).
    knee_rotation_r/l       : transverse plane, tibia rotation autour axe long
                              fémur. Positive = rotation externe du tibia.
    ankle_rotation_r/l      : transverse plane, foot rotation autour axe long
                              tibia. Positive = rotation externe du pied.
    foot_progression_r/l    : toes-out angle, plan horizontal, foot anterior
                              vs pelvis anterior. Positive = pied en dehors
                              (en abduction de la direction de marche).
    trunk_flexion           : sagittal, torso long axis vs world vertical
                              dans plan pelvis-sagittal. Positive = forward
                              lean (lean avant).
    trunk_lean_lateral      : frontal, torso vs world vertical dans plan
                              pelvis-frontal. Positive = lean vers la droite.
    trunk_rotation          : axial, torso anterior vs pelvis anterior dans
                              plan pelvis-transverse. Positive = rotation
                              à droite (torso vs pelvis).

Lecture : `_body_transforms.json` (généré par _opensim_compute_body_transforms.py).
Écriture : ajoute les colonnes au `.mot` IK (in-place modify).

OpenSim convention rappel (post floor correction) :
    World X = anterior, World Y = up (vertical), World Z = lateral (right)
    Body frame axes pour membres (femur, tibia, humerus) :
        X = anterior, Y = long axis (proximal → distal, vers bas typiquement),
        Z = lateral.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Bodies dont on a besoin des world transforms. Si un de ces bodies manque
# dans le JSON (modèle alternatif), on log un warning et on skip silencieusement
# les angles dépendant de ce body — le pipeline continue.
_REQUIRED_BODIES = (
    "pelvis", "torso",
    "femur_r", "tibia_r", "calcn_r",
    "femur_l", "tibia_l", "calcn_l",
)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _project_on_plane(vec: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Project a vector onto the plane defined by its unit normal."""
    pn = _normalize(plane_normal)
    return vec - np.dot(vec, pn) * pn


def _signed_angle(v1: np.ndarray, v2: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (in degrees) from v1 to v2 around `axis` (right-hand rule).
    Returns 0 if either vector is degenerate. Result in [-180, 180]."""
    a = np.linalg.norm(v1) * np.linalg.norm(v2)
    if a < 1e-12:
        return 0.0
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    cos_a = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(v1n, v2n), _normalize(axis)))
    return float(np.degrees(np.arctan2(sin_a, cos_a)))


# ---- Per-frame angle computations ----------------------------------------- #
#
# Convention OpenSim Rajagopal-derived (ce modèle) : pour CHAQUE body, le frame
# local est ISB-aligned avec X=anterior, Y=vertical-du-segment (= up quand le
# sujet est debout en pose neutre), Z=lateral. Donc body Y N'EST PAS l'axe
# long du bone — c'est l'axe vertical. L'axe long va de proximal (origine du
# body, situé au joint center) vers distal (Y=-0.X dans le body frame).
# Plus robuste : utiliser directement les positions des JOINT CENTRES
# (= colonne 4 des world_transforms) pour définir les vecteurs segmentaires.


def _cardan_zxy_knee(R_femur: np.ndarray, R_tibia: np.ndarray) -> tuple[float, float, float]:
    """Décompose R_rel = R_femur^T @ R_tibia en angles Cardan Z-X-Y
    (convention ISB knee : flexion autour Z, abduction autour X, rotation
    autour Y).
        R_rel = Rz(α) · Rx(β) · Ry(γ)
        sin(β) = R[2, 1]
        tan(α) = -R[0, 1] / R[1, 1]    (si cos(β) ≠ 0)
        tan(γ) = -R[2, 0] / R[2, 2]
    Retourne (α, β, γ) en degrés. Isolation clean de la rotation longitudinale
    contrairement à une projection géométrique simple qui se corrompt quand
    le femur fléchit profondément (gimbal-like singularity dans le squat).
    """
    R = R_femur.T @ R_tibia
    sin_b = float(np.clip(R[2, 1], -1.0, 1.0))
    beta = np.arcsin(sin_b)
    cos_b = np.cos(beta)
    if abs(cos_b) < 1e-6:
        # Gimbal lock : on choisit α = 0 et γ absorbe le reste
        alpha = 0.0
        gamma = float(np.arctan2(R[0, 2], R[0, 0]))
    else:
        alpha = float(np.arctan2(-R[0, 1], R[1, 1]))
        gamma = float(np.arctan2(-R[2, 0], R[2, 2]))
    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)


def _knee_valgus(femur_R: np.ndarray, tibia_R: np.ndarray, side: str) -> float:
    """Cardan Z-X-Y abduction angle = rotation autour de l'axe X (anterior)
    du fémur = abduction/adduction du tibia = clinical varus/valgus.
    Sign convention : positive = valgus (tibia inward / vers la ligne médiane)."""
    _, beta, _ = _cardan_zxy_knee(femur_R, tibia_R)
    # ISB sign convention: pour genou droit, abd vers ligne médiane = négatif.
    # On inverse pour avoir "positive = valgus" cohérent bilatéralement.
    return -beta if side == "r" else beta


def _knee_rotation(femur_R: np.ndarray, tibia_R: np.ndarray, side: str) -> float:
    """Cardan Z-X-Y rotation angle = rotation autour de l'axe Y (vertical
    segmentaire) = internal/external rotation du tibia. Sign convention :
    positive = rotation externe du tibia."""
    _, _, gamma = _cardan_zxy_knee(femur_R, tibia_R)
    return gamma if side == "r" else -gamma


def _ankle_rotation(knee_pos: np.ndarray, ankle_pos: np.ndarray,
                    tibia_R: np.ndarray, foot_R: np.ndarray,
                    side: str) -> float:
    """Transverse plane : rotation du pied autour de l'axe long du tibia.
    Axe = (ankle - knee) normalisé. Positive = rotation externe."""
    shank_axis = _normalize(ankle_pos - knee_pos)
    tibia_X_t = _project_on_plane(tibia_R[:, 0], shank_axis)
    foot_X_t = _project_on_plane(foot_R[:, 0], shank_axis)
    raw = _signed_angle(tibia_X_t, foot_X_t, -shank_axis)
    return raw if side == "r" else -raw


def _foot_progression(pelvis_R: np.ndarray, foot_R: np.ndarray,
                      side: str) -> float:
    """Toes-out : angle entre l'axe anterior du pied et celui du pelvis,
    projetés dans le plan horizontal (perpendiculaire à world Y).
    Sign convention bilatérale : POSITIVE = toes out (abduction du pied
    par rapport à la direction de marche) POUR LES DEUX CÔTÉS. On inverse
    le signe brut pour le côté droit (car rotation CW vue de dessus =
    négative en convention right-hand-rule autour de +Y world)."""
    world_Y = np.array([0.0, 1.0, 0.0])
    pelvis_X = pelvis_R[:, 0]
    foot_X = foot_R[:, 0]
    pelvis_X_h = _project_on_plane(pelvis_X, world_Y)
    foot_X_h = _project_on_plane(foot_X, world_Y)
    raw = _signed_angle(pelvis_X_h, foot_X_h, world_Y)
    return -raw if side == "r" else raw


def _trunk_flexion(pelvis_R: np.ndarray, torso_R: np.ndarray) -> float:
    """Sagittal forward lean : axe long du torse vs verticale world,
    projetés dans le plan sagittal (perpendiculaire à pelvis_Z = lateral).
    Positive = lean avant (torso bascule vers pelvis_X = anterior)."""
    world_Y = np.array([0.0, 1.0, 0.0])
    pelvis_Z = pelvis_R[:, 2]
    torso_Y = torso_R[:, 1]
    world_Y_s = _project_on_plane(world_Y, pelvis_Z)
    torso_Y_s = _project_on_plane(torso_Y, pelvis_Z)
    # axe -pelvis_Z (vers la gauche du sujet) : la convention right-hand
    # donne "positive = forward lean" quand on regarde le sujet de profil
    # droit.
    return _signed_angle(world_Y_s, torso_Y_s, -pelvis_Z)


def _trunk_lean_lateral(pelvis_R: np.ndarray, torso_R: np.ndarray) -> float:
    """Frontal lateral lean : torso_Y vs world Y dans plan frontal
    (perpendiculaire à pelvis_X = anterior). Positive = lean vers la droite
    (vers +pelvis_Z dans la convention OpenSim)."""
    world_Y = np.array([0.0, 1.0, 0.0])
    pelvis_X = pelvis_R[:, 0]
    torso_Y = torso_R[:, 1]
    world_Y_f = _project_on_plane(world_Y, pelvis_X)
    torso_Y_f = _project_on_plane(torso_Y, pelvis_X)
    # Axis -pelvis_X gives "positive = lean toward +pelvis_Z (right)".
    return _signed_angle(world_Y_f, torso_Y_f, -pelvis_X)


def _trunk_rotation(pelvis_R: np.ndarray, torso_R: np.ndarray) -> float:
    """Axial rotation tronc vs pelvis dans plan transverse (perpendiculaire
    à pelvis_Y = up). Positive = torso rotated to the right vs pelvis."""
    pelvis_Y = pelvis_R[:, 1]
    pelvis_X = pelvis_R[:, 0]
    torso_X = torso_R[:, 0]
    pelvis_X_t = _project_on_plane(pelvis_X, pelvis_Y)
    torso_X_t = _project_on_plane(torso_X, pelvis_Y)
    # -pelvis_Y : positive = rotation droite (vers +pelvis_Z)
    return _signed_angle(pelvis_X_t, torso_X_t, -pelvis_Y)


# ---- Main API ------------------------------------------------------------ #

def compute_clinical_angles(body_transforms_path: str | Path) -> dict[str, np.ndarray]:
    """Compute the 11 derived clinical angles for every frame.

    Returns: {column_name: np.ndarray of shape (n_frames,)}
    Returns {} if the body_transforms JSON is missing / unparseable.
    """
    bt_path = Path(body_transforms_path)
    if not bt_path.is_file():
        print(f"[clinical_angles] body_transforms.json missing at {bt_path}, "
              "skipping derived angles.")
        return {}
    try:
        data = json.loads(bt_path.read_text())
    except Exception as err:
        print(f"[clinical_angles] failed to parse {bt_path}: {err}")
        return {}

    n_frames = int(data.get("n_frames", 0))
    bodies = data.get("bodies", {})

    missing = [b for b in _REQUIRED_BODIES if b not in bodies]
    if missing:
        print(f"[clinical_angles] required bodies missing in body_transforms: "
              f"{missing}. Skipping derived angles.")
        return {}

    # Pré-extract rotation matrices (3x3) ET origines de body en world par
    # body × frame. Les origines (= colonne 4 des world_transforms) sont les
    # joint centres : femur_r.origin = hip joint, tibia_r.origin = knee joint,
    # calcn_r.origin = ankle-ish (calcaneus body).
    R_by_body: dict[str, np.ndarray] = {}
    pos_by_body: dict[str, np.ndarray] = {}
    for bn in _REQUIRED_BODIES:
        wts = bodies[bn]["world_transforms"]
        if len(wts) < n_frames:
            print(f"[clinical_angles] body {bn} has only {len(wts)}/{n_frames} "
                  "world transforms, skipping.")
            return {}
        wts_np = np.asarray(wts, dtype=np.float64)  # (n_frames, 4, 4)
        R_by_body[bn] = wts_np[:, :3, :3]
        pos_by_body[bn] = wts_np[:, :3, 3]

    col_names = [
        "knee_valgus_r", "knee_valgus_l",
        "knee_rotation_r", "knee_rotation_l",
        "ankle_rotation_r", "ankle_rotation_l",
        "foot_progression_r", "foot_progression_l",
        "trunk_flexion", "trunk_lean_lateral", "trunk_rotation",
    ]
    cols: dict[str, list[float]] = {n: [] for n in col_names}

    for f in range(n_frames):
        pelvis_R = R_by_body["pelvis"][f]
        torso_R = R_by_body["torso"][f]
        pelvis_X = pelvis_R[:, 0]  # anterior direction in world
        for side in ("r", "l"):
            femur_R = R_by_body[f"femur_{side}"][f]
            tibia_R = R_by_body[f"tibia_{side}"][f]
            foot_R = R_by_body[f"calcn_{side}"][f]
            hip_p = pos_by_body[f"femur_{side}"][f]
            knee_p = pos_by_body[f"tibia_{side}"][f]
            ankle_p = pos_by_body[f"calcn_{side}"][f]
            cols[f"knee_valgus_{side}"].append(
                _knee_valgus(femur_R, tibia_R, side))
            cols[f"knee_rotation_{side}"].append(
                _knee_rotation(femur_R, tibia_R, side))
            cols[f"ankle_rotation_{side}"].append(
                _ankle_rotation(knee_p, ankle_p, tibia_R, foot_R, side))
            cols[f"foot_progression_{side}"].append(
                _foot_progression(pelvis_R, foot_R, side))
        cols["trunk_flexion"].append(_trunk_flexion(pelvis_R, torso_R))
        cols["trunk_lean_lateral"].append(_trunk_lean_lateral(pelvis_R, torso_R))
        cols["trunk_rotation"].append(_trunk_rotation(pelvis_R, torso_R))

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def append_columns_to_mot(mot_path: str | Path,
                          columns: dict[str, np.ndarray]) -> int:
    """Append (in-place) columns to an OpenSim .mot file. Updates `nColumns`
    in the header, ajoute les noms à la ligne header de colonnes, ajoute
    les valeurs à chaque ligne data.

    Returns: nombre de lignes data modifiées.
    """
    if not columns:
        return 0

    mot_path = Path(mot_path)
    text = mot_path.read_text()
    lines = text.splitlines()

    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            end_idx = i
            break
    if end_idx is None or end_idx + 2 > len(lines):
        raise ValueError(f"{mot_path}: 'endheader' not found or no data lines")

    # Update <nColumns> dans le header
    for i in range(end_idx):
        low = lines[i].lower()
        if low.startswith("ncolumns="):
            try:
                old_n = int(lines[i].split("=", 1)[1].strip())
            except ValueError:
                old_n = None
            if old_n is not None:
                lines[i] = f"nColumns={old_n + len(columns)}"
            break

    # Add column names to the col-names header line (just after endheader)
    cnames_idx = end_idx + 1
    new_col_names = "\t".join(columns.keys())
    lines[cnames_idx] = lines[cnames_idx].rstrip() + "\t" + new_col_names

    # Append values to each data row
    col_arrays = list(columns.values())
    n_rows_modified = 0
    out_lines = lines[: cnames_idx + 1]
    for raw in lines[cnames_idx + 1:]:
        if not raw.strip():
            out_lines.append(raw)
            continue
        if n_rows_modified >= len(col_arrays[0]):
            # Plus de valeurs dispo (race entre n_frames JSON vs n rows .mot) :
            # on laisse les NaN/blanks et on log à la fin.
            out_lines.append(raw)
            continue
        vals = [f"{float(arr[n_rows_modified]):.6f}" for arr in col_arrays]
        out_lines.append(raw.rstrip() + "\t" + "\t".join(vals))
        n_rows_modified += 1

    mot_path.write_text("\n".join(out_lines) + "\n")
    return n_rows_modified


def add_clinical_angles_to_mot(mot_path: str | Path,
                               body_transforms_path: str | Path) -> int:
    """One-shot helper : compute + append. Returns number of angles added
    (0 si pas de body_transforms ou si bodies manquants)."""
    columns = compute_clinical_angles(body_transforms_path)
    if not columns:
        return 0
    n_modified = append_columns_to_mot(mot_path, columns)
    print(f"[clinical_angles] {len(columns)} colonnes ajoutées au .mot "
          f"({n_modified} lignes data mises à jour) : {list(columns.keys())}")
    return len(columns)
