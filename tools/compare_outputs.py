#!/usr/bin/env python3
"""Compare deux dossiers de sortie du pipeline.

Sert a prouver qu'un changement n'a rien modifie numeriquement. Sur un pipeline
qui produit des mesures cliniques, « ca a l'air pareil » ne suffit pas : on veut
l'egalite stricte, et quand elle est rompue on veut savoir OU (quel marqueur,
quelle frame), sinon on cherche a l'aveugle.

    python tools/compare_outputs.py outputs/valid_baseline outputs/valid_apres
"""
import sys
from pathlib import Path

import numpy as np


def _lire_trc(p):
    """En-tete (lignes brutes), noms de colonnes, matrice numerique."""
    lignes = p.read_text(errors="replace").splitlines()
    # Ligne 3 = noms de champs, ligne 4 = noms de marqueurs, données à partir de 6.
    entete, marqueurs = lignes[:5], lignes[3].split("\t")
    donnees = []
    for l in lignes[5:]:
        if not l.strip():
            continue
        try:
            donnees.append([float(x) if x.strip() else np.nan for x in l.split("\t")])
        except ValueError:
            continue
    largeur = max((len(r) for r in donnees), default=0)
    m = np.full((len(donnees), largeur), np.nan)
    for i, r in enumerate(donnees):
        m[i, :len(r)] = r
    return entete, [x for x in marqueurs if x.strip()], m


def _lire_sto(p):
    """Fichier OpenSim .mot/.sto : en-tete jusqu'a endheader, puis colonnes."""
    lignes = p.read_text(errors="replace").splitlines()
    i = next((k for k, l in enumerate(lignes) if l.strip().lower() == "endheader"), -1)
    if i < 0:
        return lignes[:5], [], np.empty((0, 0))
    entete, colonnes = lignes[:i + 1], lignes[i + 1].split("\t")
    donnees = []
    for l in lignes[i + 2:]:
        if not l.strip():
            continue
        try:
            donnees.append([float(x) for x in l.split()])
        except ValueError:
            continue
    return entete, colonnes, np.array(donnees) if donnees else np.empty((0, 0))


def _compare(nom, a, b, lecteur, noms_colonnes=True):
    ea, ca, ma = lecteur(a)
    eb, cb, mb = lecteur(b)
    pb = []

    if ca != cb:
        pb.append(f"    colonnes differentes : {len(ca)} vs {len(cb)}")
    if ma.shape != mb.shape:
        pb.append(f"    dimensions differentes : {ma.shape} vs {mb.shape}")
        print(f"  {nom} : DIFFERENT"); [print(x) for x in pb]; return False

    # NaN au meme endroit ? Un NaN qui apparait/disparait est un vrai signal.
    na, nb = np.isnan(ma), np.isnan(mb)
    if not np.array_equal(na, nb):
        pb.append(f"    NaN a des positions differentes ({na.sum()} vs {nb.sum()})")

    d = np.abs(np.where(na | nb, 0.0, ma - mb))
    dmax = float(d.max()) if d.size else 0.0
    if dmax == 0.0 and not pb:
        print(f"  {nom} : IDENTIQUE  ({ma.shape[0]} lignes x {ma.shape[1]} colonnes)")
        return True

    pb.append(f"    ecart absolu max : {dmax:.3e}")
    # Localiser : les 5 pires colonnes, avec la frame concernee.
    if d.size:
        par_col = d.max(axis=0)
        for j in np.argsort(par_col)[::-1][:5]:
            if par_col[j] == 0:
                break
            frame = int(np.argmax(d[:, j]))
            etiq = ca[j] if noms_colonnes and j < len(ca) else f"col{j}"
            pb.append(f"      {etiq:<24} max {par_col[j]:.3e} @ ligne {frame}")
    print(f"  {nom} : DIFFERENT")
    [print(x) for x in pb]
    return False


def main(dir_a, dir_b):
    a, b = Path(dir_a), Path(dir_b)
    print(f"A = {a}\nB = {b}\n")
    ok = True
    vus = 0
    for motif, lecteur in ((".trc", _lire_trc), ("_ik.mot", _lire_sto),
                           ("_ik_marker_errors.sto", _lire_sto)):
        for fa in sorted(a.glob(f"*{motif}")):
            fb = b / fa.name
            if not fb.exists():
                print(f"  {fa.name} : ABSENT dans B"); ok = False; continue
            vus += 1
            ok &= _compare(fa.name, fa, fb, lecteur)
    if vus == 0:
        print("  aucun artefact comparable trouve"); return 1
    print("\n=== " + ("IDENTIQUE" if ok else "DES ECARTS EXISTENT") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
