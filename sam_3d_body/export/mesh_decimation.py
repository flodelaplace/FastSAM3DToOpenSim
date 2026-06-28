"""Mesh decimation for animated GLB export (write_mesh_glb).

Approach: Farthest Point Sampling (FPS) on the template mesh to keep N
representative vertices (subset of the original 18439). For each frame the
"decimated mesh" is simply the per-frame positions of those selected vertices
— no averaging, no extra interpolation. Faces are remapped : each original
vertex → its nearest selected vertex (via cKDTree), then degenerate triangles
(2+ vertices in same representative) are dropped.

Why FPS rather than quadric edge collapse :
- Frame-coherent by construction (a selected vertex keeps its identity across
  all frames → no per-frame averaging artifacts).
- No external dependency (pure numpy + scipy.cKDTree).
- Preserves anatomical features (FPS naturally picks extremities — fingertips,
  nose tip, toes — before filling the body).

Quality is fine for kinesiology visual rendering. Topology can be a little
rough in zones of strong decimation, but silhouette stays clean. Visual
inspection on a 1.85 m squat at target=8000 (44 % of original) preserved
the global silhouette + face / hand details.
"""

from __future__ import annotations

import numpy as np


def fps_select(template_verts: np.ndarray, target_n: int) -> np.ndarray:
    """Farthest Point Sampling over the template vertices.

    Args:
        template_verts: (N, 3) reference mesh (e.g. the first valid frame).
        target_n: number of vertices to keep.

    Returns:
        selected: (target_n,) int64 array of indices into template_verts.
    """
    N = template_verts.shape[0]
    if target_n >= N:
        return np.arange(N, dtype=np.int64)

    selected = np.empty(target_n, dtype=np.int64)
    # Deterministic seed : start at vertex 0 for reproducibility.
    selected[0] = 0
    distances = np.linalg.norm(template_verts - template_verts[0], axis=1)
    for i in range(1, target_n):
        next_idx = int(np.argmax(distances))
        selected[i] = next_idx
        new_d = np.linalg.norm(template_verts - template_verts[next_idx], axis=1)
        np.minimum(distances, new_d, out=distances)
    return selected


def decimate_animated_mesh(
    template_verts: np.ndarray,
    faces: np.ndarray,
    target_n: int = 8000,
):
    """Decimate template mesh via FPS, return mapping & remapped faces.

    Args:
        template_verts: (N, 3) reference mesh (typically frame 0).
        faces: (M, 3) int triangle indices into template_verts.
        target_n: number of vertices to retain.

    Returns:
        selected (target_n,) int64 : indices of retained vertices.
        faces_new (M', 3) int32 : remapped faces (degenerate dropped).
        Per-frame projection : `decimated_verts = frame_verts[selected]`.
    """
    N = template_verts.shape[0]
    if target_n >= N:
        return np.arange(N, dtype=np.int64), faces.astype(np.int32, copy=False)

    from scipy.spatial import cKDTree

    selected = fps_select(template_verts, target_n)
    # For each original vertex, find its nearest representative (= index INTO
    # selected, in [0, target_n)).
    tree = cKDTree(template_verts[selected])
    _, vertex_to_repr = tree.query(template_verts, k=1)
    vertex_to_repr = vertex_to_repr.astype(np.int32)

    faces_new = vertex_to_repr[faces]  # (M, 3) indices in [0, target_n)
    # Drop degenerate triangles (2+ vertices collapse to same representative).
    keep = (
        (faces_new[:, 0] != faces_new[:, 1])
        & (faces_new[:, 1] != faces_new[:, 2])
        & (faces_new[:, 0] != faces_new[:, 2])
    )
    faces_new = faces_new[keep].astype(np.int32, copy=False)
    return selected, faces_new


def compress_glb_with_draco(glb_path) -> bool:
    """Post-process a GLB file with `gltf-transform optimize --compress quantize`.

    Pourquoi pas Draco/meshopt directement :
    - Draco (KHR_draco_mesh_compression) ne compresse QUE le mesh statique, pas
      les morph targets. Or 90% de la taille du fichier vient des morph targets
      (200 frames × 18439 vertices × 12 bytes ≈ 45 MB).
    - Meshopt (EXT_meshopt_compression) compresse mieux MAIS n'est pas supporté
      par Blender natif (besoin d'un addon), et la quantization de positions
      cause des misalignments vs le mesh anatomical.
    - `optimize --compress quantize` applique KHR_mesh_quantization (extension
      Khronos officielle, supportée NATIVEMENT par Blender + three.js +
      model-viewer + Babylon.js) qui quantize aussi les morph targets sur 16
      bits → ratio ~2.1× sans perte visuelle notable et sans misalignment.

    Returns True on success, False if gltf-transform is missing or fails. In
    failure case the original file is preserved untouched.
    """
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("gltf-transform") is None:
        print("  [GLB compress] gltf-transform not in PATH — skipping")
        return False
    p = Path(glb_path)
    # gltf-transform exige extension .glb/.gltf en sortie.
    tmp = p.with_name(p.stem + ".q" + p.suffix)
    try:
        res = subprocess.run(
            ["gltf-transform", "optimize",
             "--compress", "quantize", str(p), str(tmp)],
            capture_output=True, text=True, timeout=300,
        )
        if res.returncode != 0:
            print(f"  [GLB compress] gltf-transform returncode={res.returncode}")
            print(f"  [GLB compress] stderr: {res.stderr[:500]}")
            tmp.unlink(missing_ok=True)
            return False
        if not tmp.exists():
            print(f"  [GLB compress] tmp output not created: {tmp}")
            return False
        tmp.replace(p)
        return True
    except Exception as e:
        print(f"  [GLB compress] exception: {type(e).__name__}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False
