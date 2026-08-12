"""Patch a MakeHuman-exported GLB so that body / clothes / teeth materials
switch from the default Alpha BLEND to OPAQUE, while leaving the materials
that need transparency (eyebrows, eyelashes, cornea = `high-poly`) untouched.

Why: MakeHuman exports every material with `alphaMode=BLEND` by default. Most
viewers (Blender material preview, gltf-viewer.donmccurdy.com, etc.) then
render the skin / clothes translucent, which hides the avatar against the
scene. Switching to OPAQUE for opaque-by-nature materials fixes that without
having to round-trip through Blender.

Usage
-----
    python scripts/patch_avatar_opaque.py <input.glb> <output.glb>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pygltflib


# Substrings that mark a material as one that SHOULD stay alpha-blended.
KEEP_BLEND_HINTS: tuple[str, ...] = (
    "high-poly",   # MakeHuman cornea — must be transparent to see the iris
    "eyelashes",
    "eyelash",
    "eyebrow",
    "cornea",
    "iris",
)


def _base_color_has_alpha(gltf: pygltflib.GLTF2, mat) -> bool:
    """True si la texture de couleur de base est réellement découpée en alpha.

    Les cheveux MakeHuman sont des cartes planes dont la forme vient uniquement
    du canal alpha de la texture. Les forcer en OPAQUE affiche les quads pleins :
    on voit une calotte au lieu d'une coiffure. Les noms d'assets étant trop
    variés pour une liste (`afro01`, `short02`, `ponytail01`, `bob01`…), on
    regarde la texture elle-même.
    """
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return False

    pbr = getattr(mat, "pbrMetallicRoughness", None)
    tex_info = getattr(pbr, "baseColorTexture", None) if pbr else None
    if tex_info is None:
        return False
    try:
        image = gltf.images[gltf.textures[tex_info.index].source]
        if image.bufferView is None:
            return False
        view = gltf.bufferViews[image.bufferView]
        blob = gltf.binary_blob()
        offset = view.byteOffset or 0
        img = Image.open(BytesIO(blob[offset:offset + view.byteLength]))
        if img.mode not in ("RGBA", "LA", "PA"):
            return False
        alpha = img.getchannel("A")
        # Une texture opaque a un alpha constant à 255 ; une carte de cheveux
        # descend à 0 sur une large part de sa surface.
        return alpha.getextrema()[0] < 250
    except Exception:
        return False


def patch_glb_opaque(src: str | Path, dst: str | Path) -> tuple[int, int]:
    """Return (n_switched_to_opaque, n_kept_transparent)."""
    src, dst = Path(src), Path(dst)
    gltf = pygltflib.GLTF2().load(str(src))
    n_opaque = 0
    n_kept = 0
    for mat in gltf.materials:
        name_lc = (mat.name or "").lower()
        if any(h in name_lc for h in KEEP_BLEND_HINTS):
            n_kept += 1
            continue
        if _base_color_has_alpha(gltf, mat):
            n_kept += 1
            continue
        if mat.alphaMode != "OPAQUE":
            mat.alphaMode = "OPAQUE"
            if mat.alphaCutoff is not None:
                mat.alphaCutoff = None
            n_opaque += 1
    gltf.save_binary(str(dst))
    return n_opaque, n_kept


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = Path(argv[1])
    dst = Path(argv[2])
    n_opaque, n_kept = patch_glb_opaque(src, dst)
    print(f"{src.name}: {n_opaque} → OPAQUE, {n_kept} kept transparent → {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
