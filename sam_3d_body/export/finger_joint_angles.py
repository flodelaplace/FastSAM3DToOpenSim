"""Doigts pilotes par ANGLES ARTICULAIRES plutot que par directions.

Pourquoi
--------
Aligner la direction absolue de chaque phalange suppose que les deux squelettes
aient des poses de repos comparables. Elles ne le sont pas : le MHR au repos a
9,8 deg d'etalement lateral entre les doigts, l'avatar MakeHuman 28 deg. La
rotation imposee est donc ample et differente par doigt, et son roulis part de
travers.

Les tentatives de corriger cela par une reference neutre ont echoue faute d'une
reference PAR PHALANGE : le template ne fournit que la direction globale du doigt
(base -> bout), qui ne vaut pas pour la phalange proximale a cause de la courbure
naturelle au repos. C'est precisement la que le defaut restait visible.

L'angle articulaire evite le probleme entierement : l'angle entre deux segments
consecutifs est une grandeur anatomique dont le zero est le doigt tendu. Il ne
depend d'aucune pose de reference, ni cote sujet ni cote avatar. On le mesure
chez le sujet et on l'impose a l'articulation de l'avatar.

Convention : angle signe autour de l'axe X du repere main (medio-lateral), positif
en flexion vers la paume. La composante hors plan est ignoree — un doigt ne fait
pas d'abduction dans ce modele, ce qui est le comportement voulu.
"""

from __future__ import annotations

import numpy as np

# Segment parent de chaque phalange. ":hand" = axe long de la main, utilise pour
# la phalange proximale dont le parent anatomique (le metacarpien) n'est pas
# mesure par les keypoints.
_PARENT_SEG = {"1": ":hand", "2": ("MCP", "PIP"), "3": ("PIP", "DIP")}
_OWN_SEG = {"1": ("MCP", "PIP"), "2": ("PIP", "DIP"), "3": ("DIP", "TIP")}
_FINGER_OF = {"1": "Thumb", "2": "Index", "3": "Middle", "4": "Ring", "5": "Pinky"}


def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Angle signe de `a` vers `b` autour de `axis`, apres projection.

    Les deux vecteurs sont projetes dans le plan perpendiculaire a `axis` : seule
    la flexion est retenue, l'abduction est ecartee.
    """
    a = a - axis * np.sum(a * axis, axis=-1, keepdims=True)
    b = b - axis * np.sum(b * axis, axis=-1, keepdims=True)
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    a = a / (na + 1e-12)
    b = b / (nb + 1e-12)
    cos = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    sin = np.sum(np.cross(a, b) * axis, axis=-1)
    ang = np.arctan2(sin, cos)
    ang[(na[..., 0] < 1e-9) | (nb[..., 0] < 1e-9)] = 0.0
    return ang


def joint_angles(
    positions: np.ndarray,
    name_to_idx: dict[str, int],
    hand_frames: dict[str, np.ndarray],
    bones: list[str],
) -> dict[str, np.ndarray]:
    """{nom_os: angle de flexion par frame (rad)} pour les os exploitables."""
    out: dict[str, np.ndarray] = {}
    for bone in bones:
        side = bone[-1]
        finger = _FINGER_OF.get(bone[6:7])
        level = bone[8:9]
        if side not in hand_frames or finger is None or level not in _OWN_SEG:
            continue
        H = hand_frames[side]                       # (T,3,3)
        X = H[:, :, 0]                              # axe de flexion
        Y = H[:, :, 1]                              # axe long de la main

        def seg(pair):
            a, b = f"{side}{finger}{pair[0]}", f"{side}{finger}{pair[1]}"
            if a not in name_to_idx or b not in name_to_idx:
                return None
            return positions[:, name_to_idx[b], :] - positions[:, name_to_idx[a], :]

        own = seg(_OWN_SEG[level])
        if own is None:
            continue
        par = _PARENT_SEG[level]
        parent_vec = Y if par == ":hand" else seg(par)
        if parent_vec is None:
            continue
        out[bone] = _signed_angle(parent_vec, own, X)
    return out


def avatar_hand_axis(rig, side: str) -> np.ndarray | None:
    """Axe de flexion de la main de l'avatar en bind (monde), ou None."""
    n2i = rig.name_to_local_idx
    need = [f"finger{i}-1.{side}" for i in (2, 5)] + [f"wrist.{side}"]
    if any(b not in n2i for b in need):
        return None
    p = lambda b: rig.bind_world[n2i[b], :3, 3]
    x = p(f"finger5-1.{side}") - p(f"finger2-1.{side}")
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else None


def avatar_palm_normal(rig, side: str) -> np.ndarray | None:
    """Normale de la paume de l'avatar en bind (monde), ou None.

    Sert a construire un axe de flexion PAR DOIGT : perpendiculaire au doigt
    lui-meme et contenu dans le plan de la paume.
    """
    n2i = rig.name_to_local_idx
    need = [f"finger{i}-1.{side}" for i in (2, 5)] + [f"wrist.{side}"]
    if any(b not in n2i for b in need):
        return None
    p = lambda b: rig.bind_world[n2i[b], :3, 3]
    x = p(f"finger5-1.{side}") - p(f"finger2-1.{side}")
    y = (p(f"finger2-1.{side}") + p(f"finger5-1.{side}")) * 0.5 - p(f"wrist.{side}")
    z = np.cross(x, y)
    n = np.linalg.norm(z)
    return z / n if n > 1e-9 else None
