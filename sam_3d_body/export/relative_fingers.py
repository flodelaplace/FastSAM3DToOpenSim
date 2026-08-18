"""Retarget RELATIF des doigts : transfere la rotation, pas la direction.

Le probleme
-----------
Le retarget aligne la direction de chaque os sur celle mesuree chez le sujet.
C'est correct quand les deux squelettes ont des poses de repos comparables. Or
les deux mains different structurellement :

    MHR au repos            : 9,8 deg d'etalement lateral (doigts paralleles)
    avatar MakeHuman en bind: 28  deg d'etalement lateral (doigts en eventail)

Imposer la direction absolue force donc une rotation ample et DIFFERENTE pour
chaque doigt, dont le roulis — non contraint en 2 DOF — part de travers. D'ou une
rotation laterale constante, visible sur tous les doigts et dans toutes les
variantes, puisqu'aucune ne touchait a ce mecanisme.

La correction
-------------
On transfere la rotation du doigt DEPUIS SON PROPRE NEUTRE, exprimee dans le
repere de la main. L'eventail naturel de l'avatar est alors conserve et seule la
flexion reelle passe. C'est la methode standard des lors que les poses de repos
different.

Le repere de la main sert de referentiel commun : une rotation qui y est exprimee
est independante de l'orientation globale du sujet comme de l'avatar, ce qui
evite toute conversion de repere — la source d'erreur principale ici.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TEMPLATE = Path(__file__).resolve().parents[2] / "assets" / "mhr_template_markers.npz"

# Marqueurs definissant le repere de la main, par cote.
_HAND_FRAME = {
    "L": ("LFAradius", "LFAulna", "LIndex", "LPinky"),
    "R": ("RFAradius", "RFAulna", "RIndex", "RPinky"),
}


def _frame(wrist: np.ndarray, idx: np.ndarray, pky: np.ndarray) -> np.ndarray:
    """Repere orthonorme de la main : X medio-lateral, Y axe long, Z normal."""
    x = pky - idx
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    y = (idx + pky) * 0.5 - wrist
    y = y - x * np.sum(y * x, axis=-1, keepdims=True)
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)          # (..., 3, 3)


def load_neutral() -> tuple[list[str], np.ndarray] | None:
    if not _TEMPLATE.exists():
        return None
    with np.load(_TEMPLATE, allow_pickle=True) as d:
        return [str(x) for x in d["names"]], np.asarray(d["positions"], dtype=np.float64)


def neutral_hand_dirs(pairs: dict[str, tuple[str, str]]) -> dict[str, np.ndarray] | None:
    """Direction neutre de chaque os, exprimee dans le repere de la main.

    Args:
        pairs: {nom_os: (marqueur_parent, marqueur_enfant)}

    Returns:
        {nom_os: vecteur unitaire (3,) en coordonnees main}, ou None si le
        template ou les marqueurs manquent.
    """
    loaded = load_neutral()
    if loaded is None:
        return None
    names, P = loaded
    idx = {n: i for i, n in enumerate(names)}

    frames: dict[str, np.ndarray] = {}
    for side, (fr, ul, ix, pk) in _HAND_FRAME.items():
        if any(m not in idx for m in (fr, ul, ix, pk)):
            continue
        wrist = (P[idx[fr]] + P[idx[ul]]) * 0.5
        frames[side] = _frame(wrist, P[idx[ix]], P[idx[pk]])

    # Les os sont pilotes par les KEYPOINTS (LIndexMCP...), mais le template
    # neutre ne contient que les marqueurs de VERTICES (LIndex, LIndexTip...) —
    # les keypoints sortent du reseau, pas du modele de personnage, donc leur
    # pose neutre n'est pas stockee. La direction neutre d'un doigt est la meme
    # grandeur anatomique qu'on la mesure sur la peau ou sur les articulations :
    # on prend donc la reference sur la paire de vertices du doigt concerne.
    # Les trois phalanges d'un doigt partagent cette reference (elles sont
    # sensiblement colineaires au repos).
    _FINGER_OF = {"1": "Thumb", "2": "Index", "3": "Middle",
                  "4": "Ring", "5": "Pinky"}
    out: dict[str, np.ndarray] = {}
    for bone in pairs:
        side = bone[-1]
        finger = _FINGER_OF.get(bone[6:7])
        if side not in frames or finger is None:
            continue
        pm, cm = f"{side}{finger}", f"{side}{finger}Tip"
        if pm not in idx or cm not in idx:
            continue                              # pouce : pas de Tip
        v = P[idx[cm]] - P[idx[pm]]
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue
        out[bone] = frames[side].T @ (v / n)      # coordonnees main
    return out or None


def hand_frames_per_frame(
    positions: np.ndarray, name_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    """Repere de la main par frame, pour chaque cote disponible. {side: (T,3,3)}"""
    out: dict[str, np.ndarray] = {}
    for side, (fr, ul, ix, pk) in _HAND_FRAME.items():
        if any(m not in name_to_idx for m in (fr, ul, ix, pk)):
            continue
        wrist = (positions[:, name_to_idx[fr], :]
                 + positions[:, name_to_idx[ul], :]) * 0.5
        out[side] = _frame(wrist,
                           positions[:, name_to_idx[ix], :],
                           positions[:, name_to_idx[pk], :])
    return out
