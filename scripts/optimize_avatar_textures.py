"""Allège un GLB avatar en retaillant ses textures, sans toucher au rig.

Pourquoi : sur un avatar MakeHuman exporté tel quel, ~91 % du poids du GLB est
de la texture (19,9 Mo sur 21,9), et l'animation ne pèse que 0,2 Mo. Pour une
app où l'avatar est vu en pied, des diffuses en 2048² — et des DENTS en 2048² —
sont très surdimensionnées.

Deux leviers, appliqués par rôle déduit du nom de l'image :
  1. retaille (dents/langue 256², yeux 512², normal maps 512², reste 1024²) ;
  2. PNG → JPEG quand le canal alpha n'est pas réellement utilisé. L'alpha est
     INDISPENSABLE aux cheveux, cils, sourcils et cornée (leur forme vient de
     lui) : on le teste sur les pixels plutôt que de se fier au nom.

Le rig, les meshes et les animations sont recopiés à l'identique.

Usage :
    python scripts/optimize_avatar_textures.py <in.glb> <out.glb> [--max 1024]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pygltflib
from PIL import Image

# Taille max par rôle. Le rôle est deviné sur le nom de l'image, qui vient de
# l'asset MakeHuman d'origine (ex. "teeth", "short02_diffuse", "..._normal").
_ROLE_MAX = (
    (("teeth", "tongue"), 256),
    (("eyebrow", "eyelash"), 512),
    (("eye",), 512),
    (("normal", "_nm", "bump"), 512),
)

_JPEG_QUALITY = 88


def _target_size(name: str, default_max: int) -> int:
    n = (name or "").lower()
    for keys, size in _ROLE_MAX:
        if any(k in n for k in keys):
            return size
    return default_max


def _alpha_is_used(img: Image.Image) -> bool:
    """True si le canal alpha porte vraiment de l'information."""
    if img.mode not in ("RGBA", "LA", "PA"):
        return False
    return img.getchannel("A").getextrema()[0] < 250


def optimize(src: Path, dst: Path, default_max: int = 1024) -> tuple[int, int]:
    gltf = pygltflib.GLTF2().load(str(src))
    blob = gltf.binary_blob()

    # 1. Ré-encode chaque image, en mémoire.
    new_data: dict[int, bytes] = {}
    for i, image in enumerate(gltf.images or []):
        if image.bufferView is None:
            continue
        view = gltf.bufferViews[image.bufferView]
        off = view.byteOffset or 0
        raw = blob[off:off + view.byteLength]
        img = Image.open(io.BytesIO(raw))
        img.load()

        target = _target_size(image.name, default_max)
        if max(img.size) > target:
            ratio = target / max(img.size)
            img = img.resize((max(1, round(img.width * ratio)),
                              max(1, round(img.height * ratio))),
                             Image.LANCZOS)

        buf = io.BytesIO()
        if _alpha_is_used(img):
            img.convert("RGBA").save(buf, format="PNG", optimize=True)
            image.mimeType = "image/png"
        else:
            img.convert("RGB").save(buf, format="JPEG",
                                    quality=_JPEG_QUALITY, optimize=True)
            image.mimeType = "image/jpeg"
        new_data[image.bufferView] = buf.getvalue()

    # 2. Reconstruit le blob binaire : les offsets de TOUTES les bufferViews
    #    bougent dès qu'une image change de taille, il faut donc tout réécrire
    #    dans l'ordre en respectant l'alignement 4 octets du glTF.
    out = bytearray()
    for idx, view in enumerate(gltf.bufferViews):
        data = new_data.get(idx)
        if data is None:
            off = view.byteOffset or 0
            data = blob[off:off + view.byteLength]
        while len(out) % 4:
            out.append(0)
        view.byteOffset = len(out)
        view.byteLength = len(data)
        out.extend(data)
    while len(out) % 4:
        out.append(0)

    gltf.buffers[0].byteLength = len(out)
    gltf.set_binary_blob(bytes(out))
    gltf.save_binary(str(dst))
    return src.stat().st_size, dst.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max", type=int, default=1024,
                    help="taille max par défaut des textures (px)")
    a = ap.parse_args()
    before, after = optimize(Path(a.src), Path(a.dst), a.max)
    print(f"  {Path(a.src).name}: {before/1e6:.1f} Mo → {after/1e6:.1f} Mo "
          f"(-{100*(before-after)/before:.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
