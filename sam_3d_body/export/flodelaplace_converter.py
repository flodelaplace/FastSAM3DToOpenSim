"""
MHR → flodelaplace XIPH mocap markerset converter (v2).

Reads `assets/correspondence_sole.json` (produced by Mesh2Marker) which maps
each marker name to ONE MHR mesh vertex index (the anatomical landmark on the
template mesh, frame-invariant via MHR's fixed topology).

For each frame we just index `pred_vertices` at the configured indices — no
more 3-source merge (kpts + jcoords + extra vertex picks). 73 markers, all
from the mesh.

Output marker names match Model_Flodelaplace_XIPH.osim MarkerSet, including
the 19 clinical clusters added via Mesh2Marker (cuisse/jambe LFLT/LFLB/LSHN/
LTIB, bras LHTO/LHAP/LHBA/LHFR + LFRM forearm) and the xiphoid process XIPH.

The old joint-center markers (LHJC/RHJC/LKJC/RKJC/LAJC/RAJC/LEJC/REJC/c_spine1)
are intentionally absent — they were virtual JC that we now constrain via
real surface clusters (better IK).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

_ASSETS = Path(__file__).resolve().parents[2] / "assets"
_CORRESPONDENCE_JSON = _ASSETS / "correspondence_sole.json"


def load_correspondence(path: Path | str | None = None) -> dict:
    """Load the correspondence JSON. Returns the full file (markers + meta)."""
    p = Path(path) if path is not None else _CORRESPONDENCE_JSON
    with open(p) as f:
        return json.load(f)


class FlodelaplaceConverter:
    """MHR pred_vertices → flodelaplace XIPH marker array.

    Each marker is the world position of one MHR mesh vertex (index defined in
    the correspondence file). Frame-invariant because MHR topology is fixed.
    """

    def __init__(self, correspondence_path: Path | str | None = None):
        data = load_correspondence(correspondence_path)
        self._meta = {
            "schema_version": data.get("schema_version"),
            "mhr_topology_id": data.get("mhr_topology_id"),
            "opensim_model": data.get("opensim_model"),
            "marker_set": data.get("marker_set"),
        }
        # ordered list of (name, [mhr_vertex_idx, ...], opensim_body) — preserves
        # the file's marker order, which we keep stable in the TRC output.
        # Un marqueur peut référencer PLUSIEURS vertices : sa position est alors
        # le CENTROÏDE du patch (contrat de la correspondance). C'est le cas des
        # marqueurs SOLE de Mesh2Marker (12-15 vertices chacun) : moyenner un
        # patch est plus stable qu'un vertex isolé sur une surface plane.
        self._markers: List[Tuple[str, List[int], str]] = []
        for m in data["markers"]:
            verts = m["mhr_vertices"]
            if not verts:
                raise ValueError(f"Marker {m['name']!r} has empty mhr_vertices")
            self._markers.append(
                (m["name"], [int(v) for v in verts], m["opensim_body"]))

        self.marker_names: List[str] = [n for n, _, _ in self._markers]
        # vertex_indices garde le PREMIER vertex (compat : plusieurs appelants
        # attendent un entier). Le centroïde vit dans _marker_vertex_lists.
        self.vertex_indices: dict[str, int] = {n: v[0] for n, v, _ in self._markers}
        self._marker_vertex_lists: dict[str, List[int]] = {
            n: v for n, v, _ in self._markers}
        self.opensim_body: dict[str, str] = {n: b for n, _, b in self._markers}

    # ── API back-compat with the legacy converter ──────────────────────────
    # The pipeline calls extract_anatomical(verts_world) to pre-stack the
    # anatomical landmarks before the coordinate transform; then convert()
    # arranges the final marker array. We keep both methods.

    def extract_anatomical(self, vertices_3d: np.ndarray) -> np.ndarray:
        """Index the configured marker vertices out of the full MHR mesh.

        Args:
            vertices_3d: (N, 18439, 3) or (18439, 3)

        Returns:
            (N, M, 3) or (M, 3) with M = len(marker_names). Row order matches
            self.marker_names (and the future markers_array column order).
        """
        single = vertices_3d.ndim == 2
        if single:
            vertices_3d = vertices_3d[np.newaxis]
        # Cas courant (1 vertex par marqueur) : indexation vectorisée directe.
        # Sinon : centroïde du patch. On garde le chemin rapide quand tous les
        # marqueurs sont mono-vertex pour ne rien changer aux perfs existantes.
        if all(len(v) == 1 for _, v, _ in self._markers):
            idxs = np.asarray([v[0] for _, v, _ in self._markers], dtype=np.int64)
            out = vertices_3d[:, idxs, :]
        else:
            out = np.empty((vertices_3d.shape[0], len(self._markers), 3),
                           dtype=vertices_3d.dtype)
            for k, (_, verts, _) in enumerate(self._markers):
                if len(verts) == 1:
                    out[:, k, :] = vertices_3d[:, verts[0], :]
                else:
                    out[:, k, :] = vertices_3d[:, verts, :].mean(axis=1)
        return out[0] if single else out

    def convert(
        self,
        keypoints_3d: np.ndarray | None = None,   # unused (kept for API compat)
        jcoords_3d:   np.ndarray | None = None,   # unused (kept for API compat)
        anat_verts_3d: np.ndarray | None = None,  # (N, M, 3) — already extracted
                                                   # via extract_anatomical()
                                                   # then passed through the
                                                   # coordinate transformer.
    ) -> Tuple[np.ndarray, List[str]]:
        """Return (markers, marker_names).

        markers shape: (N, M, 3) or (M, 3) matching the input dim, where
        M = len(self.marker_names).

        The old signature took kpts/jcoords/anat_verts to merge 3 sources.
        We now only need anat_verts (all 73 markers come from mesh vertices).
        kpts/jcoords are accepted but unused, for backwards-compatibility with
        callers that still pass them.
        """
        if anat_verts_3d is None:
            raise ValueError(
                "anat_verts_3d is required (precompute via extract_anatomical())"
            )
        # Already in the correct (marker, 3) order — extract_anatomical returns
        # rows aligned with marker_names. Caller just passed it through the
        # coordinate transform unchanged.
        return anat_verts_3d, list(self.marker_names)

    def get_marker_names(self) -> List[str]:
        return list(self.marker_names)


# ── Back-compat shim ───────────────────────────────────────────────────────
# Older code imports `load_anatomical_vertex_indices` from this module.
# Keep a thin wrapper that returns {name: idx} for the new correspondence file.

def load_anatomical_vertex_indices() -> dict[str, int]:
    """Return {marker_name: mhr_vertex_idx} from correspondence_sole.json."""
    data = load_correspondence()
    return {m["name"]: int(m["mhr_vertices"][0]) for m in data["markers"]}
