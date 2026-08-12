"""Valide un GLB avatar avant de le déposer dans assets/avatars/.

Raison d'être : `avatar_retarget.py` ignore SILENCIEUSEMENT tout os qu'il ne
trouve pas (`if bone not in name_to_local: continue`). Un avatar au rig
incompatible ne lève donc aucune erreur — il sort simplement figé, et on ne s'en
aperçoit qu'en regardant la vidéo. Ce script transforme ce mode d'échec
silencieux en échec bruyant.

Usage :
    python scripts/check_avatar_glb.py <avatar.glb> [avatar2.glb ...]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "_ar", _ROOT / "sam_3d_body" / "export" / "avatar_retarget.py")
_ar = importlib.util.module_from_spec(_spec)
sys.modules["_ar"] = _ar
_spec.loader.exec_module(_ar)

# Matériaux qui ONT le droit de rester en alpha BLEND (cf. patch_avatar_opaque)
_MAY_BLEND = ("high-poly", "eyelash", "eyebrow", "cornea")

# L'A-pose de MakeHuman donne une envergure nettement inférieure à la taille.
# Une T-pose monterait vers ~1.0 et casserait le twist du retarget.
_APOSE_SPAN_RATIO_MAX = 0.75


def check(path: Path) -> bool:
    ok = True
    rig = _ar.load_avatar_glb(str(path))
    need = set(_ar.MAKEHUMAN_BONE_TARGETS)
    have = set(rig.joint_names)

    print(f"\n=== {path.name} ===")
    print(f"  os du rig            : {len(rig.joint_names)}")

    missing = sorted(need - have)
    print(f"  os requis présents   : {len(need & have)}/{len(need)}"
          f"{'' if not missing else '  ❌ MANQUANTS: ' + ', '.join(missing)}")
    ok &= not missing

    # Taille et pose, mesurées sur les positions de bind des os.
    pos = rig.bind_world[:, :3, 3]
    height = float(pos[:, 1].max() - pos[:, 1].min())
    span = float(pos[:, 0].max() - pos[:, 0].min())
    ratio = span / height if height else 0.0
    print(f"  hauteur (bind)       : {height:.3f} m")
    print(f"  envergure/hauteur    : {ratio:.3f}", end="")
    if ratio > _APOSE_SPAN_RATIO_MAX:
        print("  ❌ ressemble à une T-pose, pas à l'A-pose attendue")
        ok = False
    else:
        print("  ✅ A-pose")

    if not (1.2 < height < 2.2):
        print(f"  ❌ hauteur invraisemblable ({height:.3f} m) — échelle douteuse")
        ok = False

    # Transparence : une peau restée en BLEND rend l'avatar translucide.
    # Un matériau peut légitimement rester en BLEND s'il est découpé en alpha
    # (cheveux, cils, cornée) — cf. patch_avatar_opaque. On réutilise sa
    # détection plutôt que de maintenir deux listes de noms divergentes.
    _pspec = importlib.util.spec_from_file_location(
        "_pao", _ROOT / "scripts" / "patch_avatar_opaque.py")
    _pao = importlib.util.module_from_spec(_pspec)
    _pspec.loader.exec_module(_pao)

    mats = rig.gltf.materials or []
    bad = [m.name for m in mats
           if str(getattr(m, "alphaMode", "")) == "BLEND"
           and not any(h in (m.name or "").lower() for h in _MAY_BLEND)
           and not _pao._base_color_has_alpha(rig.gltf, m)]
    print(f"  matériaux            : {len(mats)}", end="")
    if bad:
        print(f"  ❌ encore en BLEND : {', '.join(bad)}")
        print("     → passer scripts/patch_avatar_opaque.py")
        ok = False
    else:
        print("  ✅ opacité correcte")

    # Tout nœud portant un mesh DOIT référencer le skin, sinon il ne suit pas
    # l'armature : le corps s'anime, les vêtements restent en pose de bind et
    # l'avatar se disloque en mouvement. Invisible sur un rendu statique, d'où
    # ce contrôle.
    unskinned = [rig.gltf.meshes[n.mesh].name
                 for n in (rig.gltf.nodes or [])
                 if n.mesh is not None and n.skin is None]
    n_meshnodes = sum(1 for n in (rig.gltf.nodes or []) if n.mesh is not None)
    print(f"  meshes liés au skin  : {n_meshnodes - len(unskinned)}/{n_meshnodes}", end="")
    if unskinned:
        print(f"  ❌ NON skinnés : {', '.join(unskinned)}")
        print("     → l'avatar se disloquera en mouvement (rig ajouté après les vêtements ?)")
        ok = False
    else:
        print("  ✅")

    meshes = [m.name for m in (rig.gltf.meshes or [])]
    print(f"  meshes               : {len(meshes)} ({', '.join(meshes[:6])}"
          f"{'…' if len(meshes) > 6 else ''})")
    if len(meshes) < 2:
        print("  ⚠️  un seul mesh — avatar probablement nu")

    print(f"  VERDICT              : {'✅ utilisable' if ok else '❌ NE PAS DÉPLOYER'}")
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    # Liste matérialisée AVANT le all() : sinon le court-circuit sauterait la
    # vérification des avatars suivant le premier échec.
    results = [check(Path(p)) for p in sys.argv[1:]]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
