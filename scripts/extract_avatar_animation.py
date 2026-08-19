"""Extrait l'animation seule d'un GLB d'avatar, sans le corps.

Pourquoi
--------
Un GLB d'avatar anime pese ~4,3 Mo, dont seulement 5,8 % d'animation : 43 % de
textures et 49 % de geometrie. Comme on livre 10 morphologies par exercice, on
renvoie donc dix fois le meme corps — 42,7 Mo par exercice, 4,5 Go pour une
banque de 105 exercices.

L'app peut faire mieux : telecharger les 10 corps UNE fois (40 Mo, cache
permanent) puis, par exercice, seulement les animations. 2,5 Mo au lieu de
42,7 Mo, et le changement d'avatar devient instantane.

Ce script produit ces clips. C'est une operation soustractive : on part du GLB
deja genere et on retire meshes, materiaux, textures et skins, en conservant la
hierarchie de noeuds et l'animation. Les NOMS DE NOEUDS sont preserves a
l'identique — c'est par eux que l'app raccroche le clip au squelette du
template, donc toute divergence casse le rendu en silence.

Il faut un clip PAR AVATAR, pas un seul par exercice : le retargeting depend de
la morphologie (l'etirement du tronc va de 1,13 a 1,30 selon le gabarit), donc
une animation n'est pas transposable d'un corps a l'autre.

Usage :
    python scripts/extract_avatar_animation.py <in.glb> <out.glb>
    python scripts/extract_avatar_animation.py --dir <dossier_exercice> <sortie>
"""

from __future__ import annotations

import argparse
import os
import sys

from pygltflib import GLTF2, Buffer, BufferView

# Octets par composant glTF, indexes par componentType.
_COMP_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
          "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _accessor_nbytes(acc) -> int:
    return _COMP_SIZE[acc.componentType] * _NCOMP[acc.type] * acc.count


def extract(src: str, dst: str) -> dict:
    g = GLTF2().load(src)
    blob = g.binary_blob()
    if not g.animations:
        raise ValueError(f"{src} ne contient aucune animation")

    # --- 1. accessors reellement references par l'animation -----------------
    used: list[int] = []
    seen: set[int] = set()
    for anim in g.animations:
        for s in anim.samplers:
            for ai in (s.input, s.output):
                if ai not in seen:
                    seen.add(ai)
                    used.append(ai)

    # --- 2. recopie compacte de leurs donnees -------------------------------
    # On reconstruit un buffer ne contenant que ces accessors. Les donnees
    # d'animation ne sont jamais entrelacees en pratique ; on refuse le cas
    # plutot que de produire un fichier silencieusement faux.
    out = bytearray()
    new_views: list[BufferView] = []
    remap: dict[int, int] = {}
    for new_idx, ai in enumerate(used):
        acc = g.accessors[ai]
        bv = g.bufferViews[acc.bufferView]
        if bv.byteStride not in (None, 0):
            raise ValueError(
                f"accessor {ai} entrelace (byteStride={bv.byteStride}) : "
                "cas non gere, l'extraction serait fausse")
        start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        n = _accessor_nbytes(acc)
        while len(out) % 4:                       # alignement glTF
            out.append(0)
        new_views.append(BufferView(buffer=0, byteOffset=len(out), byteLength=n))
        out.extend(blob[start:start + n])
        remap[ai] = new_idx

    # --- 3. ne garder que ces accessors, renumerotes ------------------------
    kept = []
    for new_idx, ai in enumerate(used):
        acc = g.accessors[ai]
        acc.bufferView = new_idx
        acc.byteOffset = 0
        kept.append(acc)
    g.accessors = kept
    g.bufferViews = new_views

    for anim in g.animations:
        for s in anim.samplers:
            s.input = remap[s.input]
            s.output = remap[s.output]

    # --- 4. retirer le corps -------------------------------------------------
    # Les noeuds sont CONSERVES tels quels : ils portent les noms d'os que
    # l'animation cible et que l'app doit retrouver dans le template.
    for node in g.nodes or []:
        node.mesh = None
        node.skin = None
    g.meshes = []
    g.materials = []
    g.textures = []
    g.images = []
    g.samplers = []
    g.skins = []

    g.buffers = [Buffer(byteLength=len(out))]
    g.set_binary_blob(bytes(out))
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    g.save(dst)

    return {
        "src_mo": os.path.getsize(src) / 1048576,
        "dst_mo": os.path.getsize(dst) / 1048576,
        "noeuds": len(g.nodes or []),
        "canaux": sum(len(a.channels) for a in g.animations),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="GLB anime, ou dossier si --dir")
    ap.add_argument("sortie", help="GLB de sortie, ou dossier si --dir")
    ap.add_argument("--dir", action="store_true",
                    help="traite tous les *.glb d'un dossier d'exercice")
    a = ap.parse_args()

    if not a.dir:
        r = extract(a.source, a.sortie)
        print(f"{os.path.basename(a.sortie)}: {r['src_mo']:.2f} → "
              f"{r['dst_mo']:.2f} Mo  ({r['noeuds']} noeuds, {r['canaux']} canaux)")
        return 0

    src_tot = dst_tot = 0.0
    n = 0
    for f in sorted(os.listdir(a.source)):
        if not f.endswith(".glb"):
            continue
        try:
            r = extract(os.path.join(a.source, f), os.path.join(a.sortie, f))
        except Exception as e:                     # un clip rate ne doit pas
            print(f"  ! {f}: {e}", file=sys.stderr)   # arreter le lot
            continue
        src_tot += r["src_mo"]
        dst_tot += r["dst_mo"]
        n += 1
        print(f"  {f:<28} {r['src_mo']:6.2f} → {r['dst_mo']:5.2f} Mo")
    if n:
        print(f"\n{n} clips : {src_tot:.1f} → {dst_tot:.1f} Mo "
              f"({src_tot / dst_tot:.1f}× plus leger)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
