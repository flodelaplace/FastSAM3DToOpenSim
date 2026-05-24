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
