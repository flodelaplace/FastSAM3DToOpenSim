"""Facteurs d'échelle par segment, calculés depuis les JOINTS du rig MHR.

Porté de `Mesh2Marker/core/src/mesh2marker/segment_scale.py` (Flo).

**Pourquoi c'est mieux que des distances entre marqueurs de peau**
Le scaling historique (`opensim_ik_runner._SCALE_MEASUREMENTS`) mesure des
distances entre marqueurs posés sur la PEAU : un sujet corpulent voit donc ses
os artificiellement allongés. Les 127 joints du rig MHR, eux, ne répondent
**qu'au bloc `scale` (28)** de la base de forme — les 45 directions
d'identité/tissu mou ne les déplacent quasiment pas. Les longueurs d'os
obtenues ici sont donc **immunisées à la corpulence par construction**.

**Ce que ce module NE fait PAS** : il ne touche jamais au `.osim`. Il produit
des facteurs destinés au ScaleSet du ScaleTool d'OpenSim, qui reste seul
responsable de scaler correctement corps, inerties, géométries et contraintes.

Les facteurs sont isotropes `(s, s, s)` et RELATIFS au template MHR (betas = 0,
`assets/mhr_template_joints.npz`) : un sujet identique au template donne 1.0.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "mhr_template_joints.npz")

# body OpenSim → (proximal, distal) dans les 127 joints du rig MHR.
# Indices validés côté Mesh2Marker contre la géométrie du rig et les keypoints mhr70.
# ⚠️ PIÈGE : cheville = 20 (D) / 4 (G), et NON 24/8 — l'« ankle » de mhr70 est en
# réalité l'orteil ; utiliser 24/8 donne un tibia de travers.
SEGMENT_SPANS: dict[str, tuple[int, int]] = {
    "femur_r": (18, 19),
    "femur_l": (2, 3),
    "tibia_r": (19, 20),
    "tibia_l": (3, 4),
    "humerus_r": (39, 40),
    "humerus_l": (75, 76),
    "ulna_r": (40, 41),
    "radius_r": (40, 41),
    "ulna_l": (76, 77),
    "radius_l": (76, 77),
    "pelvis": (2, 18),
    "torso": (35, 112),
}

# Corps sans segment propre : héritent d'un parent pour que la chaîne cinématique
# et les contraintes couplées (patellofémoral, walker_knee, mtp) restent cohérentes.
# La tête n'a pas de span fiable et ne doit PAS suivre la longueur du tronc → 1.0.
SEGMENT_SCALE_INHERIT: dict[str, str] = {
    "patella_r": "femur_r", "patella_l": "femur_l",
    "talus_r": "tibia_r", "calcn_r": "tibia_r", "toes_r": "tibia_r",
    "talus_l": "tibia_l", "calcn_l": "tibia_l", "toes_l": "tibia_l",
    "sacrum": "pelvis",
    "hand_r": "radius_r", "hand_l": "radius_l",
}


def load_template_joints(path: str | None = None) -> np.ndarray:
    """Joints du template MHR (betas = 0), (127, 3) en mètres."""
    p = path or _TEMPLATE_PATH
    with np.load(p, allow_pickle=True) as d:
        return np.asarray(d["J0"], dtype=np.float64)


def subject_joints_from_frames(
    frames_joint_coords: Sequence[np.ndarray | None],
    min_frames: int = 3,
) -> np.ndarray | None:
    """Joints "sujet" robustes à partir des joints par frame de l'inférence.

    Les longueurs d'os sont invariantes à la pose (segments rigides), donc on
    prend la MÉDIANE par joint sur toutes les frames valides : ça absorbe le
    bruit d'inférence sans supposer une pose particulière.
    Retourne (127, 3) ou None si trop peu de frames exploitables.
    """
    valid = [np.asarray(j, dtype=np.float64) for j in frames_joint_coords
             if j is not None and np.all(np.isfinite(j))]
    if len(valid) < min_frames:
        return None
    return np.median(np.stack(valid, axis=0), axis=0)


def _span_scale(j0: np.ndarray, js: np.ndarray, prox: int, dist: int) -> float:
    len0 = float(np.linalg.norm(j0[dist] - j0[prox]))
    lens = float(np.linalg.norm(js[dist] - js[prox]))
    return lens / len0 if len0 > 1e-9 else 1.0


def segment_scales_from_joints(
    j_subject: np.ndarray,
    j_template: np.ndarray | None = None,
    bodies: Iterable[str] | None = None,
    clip: tuple[float, float] = (0.5, 2.0),
) -> dict[str, tuple[float, float, float]]:
    """Facteurs isotropes par corps OpenSim, à partir des joints du rig MHR.

    Args:
        j_subject  : (127, 3) joints du sujet (médiane sur les frames).
        j_template : (127, 3) joints template ; chargés depuis les assets si None.
        bodies     : si fourni, TOUS ces corps sont présents en sortie — ceux sans
                     span ni héritage reçoivent explicitement (1,1,1), ce qui permet
                     de distinguer « non scalé » de « oublié ».
        clip       : garde-fou sur des ratios aberrants (inférence dégradée).
    """
    j0 = load_template_joints() if j_template is None else np.asarray(j_template, float)
    js = np.asarray(j_subject, dtype=np.float64)
    n = min(j0.shape[0], js.shape[0])

    computed: dict[str, float] = {}
    for body, (prox, dist) in SEGMENT_SPANS.items():
        if 0 <= prox < n and 0 <= dist < n:
            computed[body] = float(np.clip(_span_scale(j0, js, prox, dist), *clip))
    for child, parent in SEGMENT_SCALE_INHERIT.items():
        if parent in computed:
            computed[child] = computed[parent]

    names = list(bodies) if bodies is not None else list(computed)
    return {name: (computed.get(name, 1.0),) * 3 for name in names}
