"""Convertit une animation par morph targets en animation par squelette.

POURQUOI. Le GLB du maillage corporel stocke la position des 18 439 sommets
pour CHAQUE image : 216 kio par frame, soit 7 Mo par seconde de vidéo. Un
exercice de trente secondes pèserait 200 Mo. Le poids est proportionnel à la
durée, ce qui n'est pas une inefficacité mais une absence de plafond.

Un maillage animé par squelette stocke la géométrie UNE fois, puis quelques
kilo-octets de transformations articulaires par image. Le poids devient
indépendant de la durée.

POURQUOI C'EST EXACT ICI, et pas une approximation. Deux conditions, toutes
deux vérifiées en production :

  - le shape-lock fige la morphologie sur toute la vidéo (médiane des
    paramètres, `demo_video_opensim.py`), donc la forme du corps appartient à
    la pose de référence et ne varie pas d'une image à l'autre ;
  - `MHR_NO_CORRECTIVES=1` est posé dans la définition de job AWS et dans le
    compose local, donc il n'y a aucune déformation dépendante de la posture
    par-dessus le skinning.

Après ça, la seule chose qui varie image par image est la pose du squelette —
et c'est très exactement ce qu'un skinning représente.

CE QUI EST DEMONTRE, ET CE QUI NE MARCHE PAS.

Ce module retrouve les transformations articulaires depuis les sommets, par
moindres carrés, sans rien changer en amont. Mesuré sur un CMJ reel de 84
images :

  - avec la frame 0 comme pose de reference : 0,058 mm de moyenne ;
  - avec la VRAIE pose de repos MHR, reconstruite ici par `pose_de_repos()` :
    **0,000 mm**. Exact au flottant pres.

C'est la preuve que le maillage EST representable par un squelette, sans perte.
La piste est bonne et la donnee s'y prete.

MAIS le refit ne suffit pas pour exporter en glTF, et c'est le point d'arret.
Une animation glTF n'anime que translation, rotation et echelle. Or le systeme
est SOUS-DETERMINE par articulation : plusieurs matrices affines differentes
produisent exactement le meme melange de sommets. Le moindres carres en choisit
une valide, mais pas celle qui est une similitude — et la decomposer fait
remonter l'erreur a 1,5 mm de moyenne et 90 mm de maximum.

La vraie transformation, elle, EST une similitude : `mhr_head` sort un
`skel_state` de forme (B, 127, 8) = translation(3) + quaternion(4) + echelle(1),
qui rentre exactement dans une animation glTF.

CONCLUSION. Rededuire les transformations depuis les sommets revient a resoudre
un probleme mal pose alors que la reponse exacte existe deja en amont. Le chemin
propre est de faire descendre `skel_state` de l'inference jusqu'a l'exportateur.
Ce module reste utile pour deux choses : la reconstruction de la pose de repos,
et la mesure de fidelite qui servira de garde-fou a l'export skinne.

GARDE-FOU. L'erreur est mesurée sur chaque fichier produit. Au-delà du seuil,
l'appelant doit retomber sur les morph targets — mieux vaut un fichier lourd
qu'un maillage faux qui ne se signale pas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Chemin des poids de skinning, extraits une fois de mhr_model.pt.
POIDS_DEFAUT = "checkpoints/sam-3d-body-dinov3/assets/_skin_mhr.npz"
_PREFIXE = "character_torch__linear_blend_skinning__"

# Garde-fou : au-dela, on refuse le skinning et on garde les morph targets.
#
# On juge sur le p99,9 et non sur le maximum absolu. Mesure sur un CMJ reel :
# moyenne 0,058 mm, p99,9 = 1,6 mm, max 4,0 mm — et 99,67 % des sommets sous le
# millimetre. Les quelques dizaines de sommets a 4 mm sont dans l'entrejambe,
# la ou le lissage Butterworth applique en espace-sommets n'est pas exactement
# representable par un skinning. Laisser un seul sommet aberrant annuler un
# gain de 17x serait absurde.
#
# Le maximum garde quand meme un plafond, tres large : un ajustement qui aurait
# vraiment echoue exploserait partout, pas sur 89 sommets.
SEUIL_P999_M = 0.003
SEUIL_MAX_M = 0.010


class DonneesSkinning:
    """Poids de skinning MHR, sous une forme exploitable pour glTF."""

    def __init__(self, chemin: str | Path):
        d = np.load(str(chemin))
        self.vert_idx = d[_PREFIXE + "vert_indices_flattened"].astype(np.int64)
        self.joint_idx = d[_PREFIXE + "skin_indices_flattened"].astype(np.int64)
        self.poids = d[_PREFIXE + "skin_weights_flattened"].astype(np.float64)
        self.n_verts = int(self.vert_idx.max()) + 1
        self.n_joints = int(self.joint_idx.max()) + 1

    def tables_gltf(self) -> tuple[np.ndarray, np.ndarray]:
        """(JOINTS_0, WEIGHTS_0) en VEC4, comme l'exige glTF.

        Le maximum d'influences est de 4 dans MHR — verifie : 3925 sommets a 1,
        3491 a 2, 3662 a 3, 7361 a 4. Le format VEC4 est donc un contenant
        EXACT, pas une troncature.
        """
        J = np.zeros((self.n_verts, 4), dtype=np.uint16)
        W = np.zeros((self.n_verts, 4), dtype=np.float32)
        rang = np.zeros(self.n_verts, dtype=np.int64)
        for v, j, w in zip(self.vert_idx, self.joint_idx, self.poids):
            r = rang[v]
            if r < 4:
                J[v, r], W[v, r] = j, w
                rang[v] = r + 1
        # Renormalisation defensive : si une somme derivait, le maillage
        # se retrecirait silencieusement a l'affichage.
        s = W.sum(axis=1, keepdims=True)
        W = np.where(s > 0, W / np.maximum(s, 1e-9), W)
        return J, W


def _matrice_normale(sk: DonneesSkinning, repos: np.ndarray):
    """Prépare les équations normales du moindres carrés.

    Inconnues : 12 paramètres par articulation (rotation 3x3 + translation).
    Le système est identique à chaque image — seul le second membre change —
    donc on le factorise UNE fois pour les N frames.
    """
    n_par = sk.n_joints * 12
    AtA = np.zeros((n_par, n_par), dtype=np.float64)
    # Pour chaque sommet, la contribution est w * [x, y, z, 1] sur les colonnes
    # de son articulation, repetee sur les trois axes.
    base = np.concatenate([repos[sk.vert_idx], np.ones((len(sk.vert_idx), 1))], axis=1)
    contrib = base * sk.poids[:, None]                      # (E, 4)

    # Regroupement par sommet : les articulations qui influencent un meme
    # sommet sont couplees dans les equations normales.
    ordre = np.argsort(sk.vert_idx, kind="stable")
    vi = sk.vert_idx[ordre]
    ji = sk.joint_idx[ordre]
    co = contrib[ordre]
    debuts = np.searchsorted(vi, np.arange(sk.n_verts))
    fins = np.searchsorted(vi, np.arange(sk.n_verts), side="right")

    for v in range(sk.n_verts):
        a, b = debuts[v], fins[v]
        js, cs = ji[a:b], co[a:b]
        for p in range(len(js)):
            for q in range(len(js)):
                bloc = np.outer(cs[p], cs[q])               # (4, 4)
                for axe in range(3):
                    r0 = js[p] * 12 + axe * 4
                    c0 = js[q] * 12 + axe * 4
                    AtA[r0:r0 + 4, c0:c0 + 4] += bloc
    return AtA, co, ji, vi, ordre


def ajuster(verts_par_frame: list[np.ndarray], sk: DonneesSkinning,
            repos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Retrouve les transformations articulaires reproduisant chaque image.

    Renvoie (repos, transformations (F, J, 3, 4), diagnostic).
    """
    frames = [v for v in verts_par_frame if v is not None]
    if not frames:
        raise ValueError("aucune frame exploitable")
    if repos is None:
        repos = np.asarray(frames[0], dtype=np.float64)

    AtA, co, ji, vi, ordre = _matrice_normale(sk, repos)
    # Regularisation de Tikhonov : certaines articulations (doigts, peu
    # influents) sont mal contraintes et la matrice serait quasi singuliere.
    AtA += np.eye(AtA.shape[0]) * 1e-6
    facteur = np.linalg.cholesky(AtA)

    poids_tries = sk.poids[ordre]
    h = np.concatenate([repos, np.ones((len(repos), 1))], axis=1)   # (V, 4)
    h_e = h[vi]                                                     # (E, 4)

    sorties, erreurs = [], []
    for V in frames:
        V = np.asarray(V, dtype=np.float64)
        cible = V[vi]                                               # (E, 3)

        Atb = np.zeros((sk.n_joints, 3, 4))
        for axe in range(3):
            np.add.at(Atb, (ji, axe), co * cible[:, axe:axe + 1])

        x = np.linalg.solve(facteur.T, np.linalg.solve(facteur, Atb.ravel()))
        M = x.reshape(sk.n_joints, 3, 4)
        sorties.append(M)

        # Reconstruction vectorisee : chaque couple sommet-articulation
        # contribue w * (M_j @ [x, y, z, 1]), somme par sommet.
        contrib = poids_tries[:, None] * np.einsum("eij,ej->ei", M[ji], h_e)
        rec = np.zeros_like(V)
        np.add.at(rec, vi, contrib)
        # Distance euclidienne par sommet, pas ecart par coordonnee : c'est
        # ce qu'un oeil verrait, et c'est ce qui se compare a une taille en mm.
        erreurs.append(np.linalg.norm(rec - V, axis=1))

    # Trois chiffres, parce qu'ils ne disent pas la meme chose : le max est
    # sensible a un seul sommet aberrant, la moyenne noie les defauts locaux,
    # le p99,9 est le compromis lisible. C'est le MAX qui sert de garde-fou.
    tous = np.concatenate(erreurs)
    diag = {"erreur_max_m": float(tous.max()),
            "erreur_moyenne_m": float(tous.mean()),
            "erreur_p999_m": float(np.percentile(tous, 99.9)),
            "pct_sous_1mm": float((tous < 0.001).mean() * 100.0),
            "n_frames": len(frames), "n_joints": sk.n_joints}
    return repos, np.stack(sorties), diag


def ajustement_acceptable(diag: dict) -> tuple[bool, str]:
    """Le skinning est-il fidèle assez pour remplacer les morph targets ?

    L'appelant DOIT consulter cette fonction et retomber sur les morph targets
    si elle refuse. Un maillage faux qui ne se signale pas est bien pire qu'un
    fichier lourd.
    """
    p999, mx = diag["erreur_p999_m"], diag["erreur_max_m"]
    if p999 > SEUIL_P999_M:
        return False, (f"p99,9 = {p999 * 1000:.2f} mm au-dessus du seuil de "
                       f"{SEUIL_P999_M * 1000:.0f} mm")
    if mx > SEUIL_MAX_M:
        return False, (f"ecart maximal = {mx * 1000:.2f} mm au-dessus du plafond "
                       f"de {SEUIL_MAX_M * 1000:.0f} mm")
    return True, (f"moyenne {diag['erreur_moyenne_m'] * 1000:.3f} mm, "
                  f"p99,9 {p999 * 1000:.2f} mm, max {mx * 1000:.2f} mm, "
                  f"{diag['pct_sous_1mm']:.1f} % des sommets sous 1 mm")


def pose_de_repos(chemin_betas: str | Path,
                  chemin_poids: str | Path = POIDS_DEFAUT) -> np.ndarray:
    """Reconstruit la pose de repos du sujet, en metres.

    C'est `base_shape` plus les 45 vecteurs de forme ponderes par les betas que
    le shape-lock a sauvegardes dans `_shape_betas.npz`. Le modele MHR travaille
    en CENTIMETRES — verifie : la pose reconstruite mesure 173,7 de haut pour
    130,6 d'envergure, bras ecartes. D'ou la division par 100.

    C'est cette pose, et non la premiere image, qui rend l'ajustement exact :
    0,000 mm contre 0,058 mm en partant de la frame 0.
    """
    d = np.load(str(chemin_poids))
    base = d["character_torch__blend_shape__base_shape"].astype(np.float64)
    vecteurs = d["character_torch__blend_shape__shape_vectors"].astype(np.float64)
    b = np.load(str(chemin_betas))
    betas = b["shape_params"] if "shape_params" in b.files else b["betas73"][:45]
    betas = np.asarray(betas, dtype=np.float64)[:vecteurs.shape[0]]
    return (base + np.tensordot(betas, vecteurs, axes=(0, 0))) / 100.0
