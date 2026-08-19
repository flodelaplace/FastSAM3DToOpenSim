"""Recolle un clip d'animation sur un template d'avatar, en glTF pur.

Pourquoi
--------
`extract_avatar_animation.py` separe le corps du geste pour alleger l'app, qui
les recombine a l'execution. Ce script fait l'operation inverse, hors ligne.

Deux usages :

  1. Controle visuel. Un clip seul ne s'ouvre pas dans Blender — il n'a pas de
     corps. Recolle sur son template, il redevient un avatar anime qu'on peut
     inspecter, ce qui est la seule facon serieuse de valider que la separation
     n'a rien casse.

  2. Repli. Si le moteur de l'app ne sait pas assembler a l'execution, on livre
     des GLB fusionnes produits par ce script, sans rien regenerer.

Le raccord se fait PAR NOM DE NOEUD, comme le ferait un moteur 3D : c'est
exactement la meme condition de validite, donc un echec ici revele un echec
cote app.

Usage :
    python scripts/merge_avatar_animation.py <template.glb> <clip.glb> <out.glb>
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

from pygltflib import GLTF2


def merge(template_path: str, clip_path: str, out_path: str) -> dict:
    tpl = GLTF2().load(template_path)
    clip = GLTF2().load(clip_path)
    tb = bytearray(tpl.binary_blob())
    cb = clip.binary_blob()

    if not clip.animations:
        raise ValueError(f"{clip_path} ne contient aucune animation")

    # --- correspondance des noeuds, par nom ---------------------------------
    tpl_idx = {n.name: i for i, n in enumerate(tpl.nodes or []) if n.name}
    manquants = set()

    # --- recopie des bufferViews du clip a la fin du buffer du template -----
    bv_offset = len(tpl.bufferViews or [])
    for bv in clip.bufferViews or []:
        while len(tb) % 4:                      # alignement glTF
            tb.append(0)
        start = bv.byteOffset or 0
        new_bv = copy.deepcopy(bv)
        new_bv.buffer = 0
        new_bv.byteOffset = len(tb)
        tb.extend(cb[start:start + bv.byteLength])
        tpl.bufferViews.append(new_bv)

    acc_offset = len(tpl.accessors or [])
    for acc in clip.accessors or []:
        new_acc = copy.deepcopy(acc)
        new_acc.bufferView = acc.bufferView + bv_offset
        tpl.accessors.append(new_acc)

    # --- report des animations ----------------------------------------------
    n_canaux = 0
    for anim in clip.animations:
        new_anim = copy.deepcopy(anim)
        for s in new_anim.samplers:
            s.input += acc_offset
            s.output += acc_offset
        gardes = []
        for ch in new_anim.channels:
            src = ch.target.node
            if src is None:
                continue
            nom = (clip.nodes[src].name if clip.nodes else None)
            if nom is None or nom not in tpl_idx:
                manquants.add(nom or f"<index {src}>")
                continue                        # piste orpheline : on l'ecarte
            ch.target.node = tpl_idx[nom]
            gardes.append(ch)
        new_anim.channels = gardes
        n_canaux += len(gardes)
        if not tpl.animations:
            tpl.animations = []
        tpl.animations.append(new_anim)

    tpl.buffers[0].byteLength = len(tb)
    tpl.set_binary_blob(bytes(tb))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tpl.save(out_path)

    return {
        "canaux": n_canaux,
        "orphelins": sorted(manquants),
        "mo": os.path.getsize(out_path) / 1048576,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("template")
    ap.add_argument("clip")
    ap.add_argument("sortie")
    a = ap.parse_args()

    r = merge(a.template, a.clip, a.sortie)
    print(f"{os.path.basename(a.sortie)} : {r['mo']:.2f} Mo, "
          f"{r['canaux']} canaux raccordes")
    if r["orphelins"]:
        print(f"  ATTENTION {len(r['orphelins'])} pistes sans cible dans le "
              f"template : {r['orphelins'][:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
