"""Injecte les articulations de main de SAM3D dans le tableau de marqueurs.

Pourquoi
--------
Le retarget pilotait les doigts avec des marqueurs de PEAU (vertices du mesh), et
pour le majeur et l'annulaire avec des positions carrement INTERPOLEES entre
l'index et l'auriculaire — 23,5 mm d'ecart mesure avec la vraie position sur une
main dont les doigts sont espaces de 15-20 mm.

Or SAM3D produit deja les articulations nommees : `mhr70.py` expose
`right-middle-first-joint`, `left-ring-second-joint`, etc. — 4 points par doigt
(bout + 3 articulations), soit 40 points de main. Ce sont des CENTRES
ARTICULAIRES, ce qui convient bien mieux a l'orientation d'un os qu'un point de
peau, et ils sont disponibles sans rien recalculer.

Convention de nommage produite : `<L|R><Doigt><Niveau>`, ex. `LMiddleMCP`,
`LMiddlePIP`, `LMiddleDIP`, `LMiddleTIP`. Les noms existants du markerset
(LMiddle, LMiddleTip…) ne sont PAS ecrases : on ajoute, on ne remplace pas.
"""

from __future__ import annotations

import numpy as np

# Indices dans les 70 keypoints MHR (cf. sam_3d_body/metadata/mhr70.py).
# L'ordre y est : tip, first-joint, second-joint, third-joint. "first" est le
# plus DISTAL (juste avant le bout), "third" le plus proximal — d'ou le mapping
# vers DIP / PIP / MCP a rebours.
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_LEVELS = ("TIP", "DIP", "PIP", "MCP")   # ordre des 4 indices consecutifs

_HAND_BASE = {"R": 21, "L": 42}          # index du premier keypoint de la main


def finger_marker_table() -> dict[str, int]:
    """{nom_de_marqueur: index_keypoint} pour les 40 articulations de main."""
    out: dict[str, int] = {}
    for side, base in _HAND_BASE.items():
        for f, finger in enumerate(_FINGERS):
            for lvl, level in enumerate(_LEVELS):
                out[f"{side}{finger}{level}"] = base + f * 4 + lvl
    return out


def append_hand_keypoints(
    markers: np.ndarray,
    marker_names: list[str],
    kpts: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Ajoute les articulations de main au tableau de marqueurs.

    Args:
        markers: (T, M, 3) marqueurs deja construits, MEME repere que kpts.
        marker_names: noms correspondants.
        kpts: (T, >=61, 3) keypoints MHR apres transformation OpenSim.

    Returns:
        (markers etendus, noms etendus). Entrees renvoyees telles quelles si les
        keypoints n'ont pas la main (modele tronque).
    """
    if kpts is None or kpts.ndim != 3 or kpts.shape[1] < 61:
        return markers, marker_names
    if markers.shape[0] != kpts.shape[0]:
        return markers, marker_names

    table = finger_marker_table()
    new_names = [n for n in table if n not in marker_names]
    if not new_names:
        return markers, marker_names

    extra = np.stack([kpts[:, table[n], :] for n in new_names], axis=1)
    return np.concatenate([markers, extra], axis=1), list(marker_names) + new_names
