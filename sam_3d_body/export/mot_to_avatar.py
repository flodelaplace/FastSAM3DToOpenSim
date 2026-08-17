"""Pilote des os de l'avatar depuis les angles articulaires de l'IK (.mot).

MAQUETTE — limitée à la cheville, pour valider la table d'axes avant d'étendre.

Pourquoi
--------
Le retarget actuel déduit l'orientation de chaque os de DEUX marqueurs. Or deux
points donnent une direction, pas un roulis : la rotation autour de l'axe de l'os
reste indéterminée et doit être réinventée os par os avec une référence axiale.
C'est la racine commune de trois défauts constatés — rachis vrillé en décubitus,
pied faux à quatre pattes, main molle.

L'IK OpenSim, elle, résout le problème par construction : `ankle_angle` et
`subtalar_angle` sont des degrés de liberté EXPLICITES et bornés par le modèle.
Mesuré sur un bird dog : 0,31-0,49°/frame de variation contre 1,23°/frame pour la
meilleure reconstruction par marqueurs, et des amplitudes dans les plages
physiologiques.

Et surtout, `local_quat` de l'avatar est déjà exprimé RELATIVEMENT AU PARENT,
tout comme un angle articulaire : la correspondance est directe, sans passer par
le monde.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Colonnes du .mot pilotant chaque os. (dorsiflexion, inversion/éversion)
_ANKLE_COLUMNS = {
    "foot.L": ("ankle_angle_l", "subtalar_angle_l"),
    "foot.R": ("ankle_angle_r", "subtalar_angle_r"),
}


def load_mot(path) -> tuple[list[str], np.ndarray]:
    """Lit un .mot OpenSim → (noms de colonnes, données (T, C))."""
    lines = open(path).read().splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[end + 1].split("\t")
    data = np.array([[float(x) for x in l.split("\t")]
                     for l in lines[end + 2:] if l.strip()])
    return cols, data


def _resample(v: np.ndarray, n_out: int) -> np.ndarray:
    """Ré-échantillonne linéairement une colonne sur n_out frames.

    L'IK et le retarget peuvent ne pas avoir le même nombre d'images (l'IK peut
    sauter des frames non résolues) : on aligne sur l'axe temporel normalisé.
    """
    if len(v) == n_out:
        return v
    return np.interp(np.linspace(0.0, 1.0, n_out),
                     np.linspace(0.0, 1.0, len(v)), v)


def apply_ankle_from_mot(rig, result, mot_path, *, verbose: bool = True) -> int:
    """Réécrit les quaternions locaux des pieds depuis le .mot. Retourne le nb d'os traités."""
    cols, data = load_mot(mot_path)
    n_done = 0

    for bone, (c_dorsi, c_inv) in _ANKLE_COLUMNS.items():
        if bone not in rig.name_to_local_idx:
            continue
        if c_dorsi not in cols or c_inv not in cols:
            continue
        ji = rig.name_to_local_idx[bone]

        # Axes anatomiques exprimés dans le repère LOCAL de l'os.
        # En pose de bind (A-pose) l'axe médio-latéral du sujet est le X monde,
        # et l'axe longitudinal du pied est la direction de l'os lui-même.
        bind_rot = rig.bind_world[ji, :3, :3]
        kids = [k for k, p in rig.joint_to_parent.items() if p == ji]
        if not kids:
            continue
        long_world = (rig.bind_world[kids[0], :3, 3] - rig.bind_world[ji, :3, 3])
        long_world /= np.linalg.norm(long_world) or 1.0
        ml_world = np.array([1.0, 0.0, 0.0])
        # orthogonalise l'axe médio-latéral contre l'axe long : les deux
        # rotations doivent être indépendantes, sinon dorsiflexion et inversion
        # se contaminent.
        ml_world = ml_world - long_world * (ml_world @ long_world)
        ml_world /= np.linalg.norm(ml_world) or 1.0

        ml_local = bind_rot.T @ ml_world
        long_local = bind_rot.T @ long_world

        dorsi = np.radians(_resample(data[:, cols.index(c_dorsi)], result.n_frames))
        inver = np.radians(_resample(data[:, cols.index(c_inv)], result.n_frames))

        bind_q = rig.bind_local_q[ji]
        for t in range(result.n_frames):
            rot = (R.from_rotvec(ml_local * dorsi[t])
                   * R.from_rotvec(long_local * inver[t]))
            q = (R.from_quat(bind_q) * rot).as_quat()
            result.local_quat[t, ji] = q
        n_done += 1
        if verbose:
            print(f"  [mot→avatar] {bone} ← {c_dorsi} "
                  f"[{np.degrees(dorsi).min():+.1f}, {np.degrees(dorsi).max():+.1f}]° "
                  f"+ {c_inv} [{np.degrees(inver).min():+.1f}, "
                  f"{np.degrees(inver).max():+.1f}]°")
    return n_done
