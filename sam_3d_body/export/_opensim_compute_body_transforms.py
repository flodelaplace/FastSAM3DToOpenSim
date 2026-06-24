"""Subprocess script — runs in the opensim conda env.

Reads a scaled .osim model and a .mot motion file, computes the world-space
4x4 transform of each Body at every frame using OpenSim's forward kinematics,
and writes the result + per-body mesh metadata as JSON.

Invoked by sam_3d_body/export/opensim_anatomical.py as:
    {opensim_python} this_script.py <osim_path> <mot_path> <out_json>
"""
from __future__ import annotations

import json
import os
import sys
import xml.dom.minidom as minidom
from pathlib import Path

import numpy as np
import opensim as osim


def _read_body_meshes(osim_path: str) -> dict[str, list[dict]]:
    """Parse .osim XML to extract per-body Mesh attachments.

    Returns: {body_name: [{file, scale, translation, orientation_xyz}, ...]}
    Translation/orientation are the LOCAL transforms of the mesh inside the body frame.

    Two valid attachment paths in OpenSim 4.x .osim files:

      (a) DIRECT:  <Body><attached_geometry><Mesh>...                → identity offset
      (b) NESTED:  <Body><components><PhysicalOffsetFrame translation>...
                       <attached_geometry><Mesh>...                  → use POF offset

    Path (b) is how the torso carries its 20 vertebra/rib/scapula meshes
    (cerv1-7, thoracic1-12, hat_ribs_scap) in flodelaplace_mocap.osim. Reading
    only path (a) collapses them to the torso origin → visually missing upper
    spine + cervical, and shoulder joint offsets look wrong because the scapula
    mesh is in the wrong place.
    """
    doc = minidom.parse(osim_path)
    out: dict[str, list[dict]] = {}
    body_set = doc.getElementsByTagName("BodySet")
    if not body_set:
        return out

    def _read_offset(el):
        """Read child <translation>/<orientation> of an element. Identity if absent."""
        tr = [0.0, 0.0, 0.0]
        ro = [0.0, 0.0, 0.0]
        if el is None:
            return tr, ro
        for child in el.childNodes:
            if child.nodeName == "translation" and child.firstChild:
                tr = [float(x) for x in child.firstChild.nodeValue.split()]
            elif child.nodeName == "orientation" and child.firstChild:
                ro = [float(x) for x in child.firstChild.nodeValue.split()]
        return tr, ro

    def _parse_meshes(attached_el, offset_tr, offset_ro):
        """Extract Mesh entries below an <attached_geometry> with a given local offset."""
        result = []
        if attached_el is None:
            return result
        for mesh in attached_el.childNodes:
            if mesh.nodeName != "Mesh":
                continue
            mf_list = mesh.getElementsByTagName("mesh_file")
            if not mf_list or not mf_list[0].firstChild:
                continue
            mesh_file = mf_list[0].firstChild.nodeValue.strip()
            scale = [1.0, 1.0, 1.0]
            sf = mesh.getElementsByTagName("scale_factors")
            if sf and sf[0].firstChild:
                scale = [float(x) for x in sf[0].firstChild.nodeValue.split()]
            result.append({
                "file": mesh_file,
                "scale": scale,
                "translation": list(offset_tr),
                "orientation_xyz": list(offset_ro),
            })
        return result

    for body in body_set[0].getElementsByTagName("Body"):
        name = body.getAttribute("name")
        meshes: list[dict] = []

        # Path (a): Body/attached_geometry → identity offset
        for child in body.childNodes:
            if child.nodeName == "attached_geometry":
                meshes.extend(_parse_meshes(child, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))

        # Path (b): Body/components/PhysicalOffsetFrame/attached_geometry
        # Each POF carries its own offset; meshes inside inherit it.
        for child in body.childNodes:
            if child.nodeName != "components":
                continue
            for pof in child.childNodes:
                if pof.nodeName != "PhysicalOffsetFrame":
                    continue
                pof_tr, pof_ro = _read_offset(pof)
                for ag in pof.childNodes:
                    if ag.nodeName != "attached_geometry":
                        continue
                    meshes.extend(_parse_meshes(ag, pof_tr, pof_ro))

        if meshes:
            out[name] = meshes
    return out


def _eulerXYZ_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from XYZ Euler angles (radians), R = Rx * Ry * Rz."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def main() -> None:
    if len(sys.argv) < 4:
        print("usage: compute_body_transforms.py <osim> <mot> <out_json>", file=sys.stderr)
        sys.exit(2)
    osim_path = sys.argv[1]
    mot_path = sys.argv[2]
    out_json = sys.argv[3]

    if not os.path.isfile(osim_path):
        raise FileNotFoundError(osim_path)
    if not os.path.isfile(mot_path):
        raise FileNotFoundError(mot_path)

    body_meshes = _read_body_meshes(osim_path)

    model = osim.Model(osim_path)
    motion = osim.TimeSeriesTable(mot_path)
    coord_set = model.getCoordinateSet()
    body_set = model.getBodySet()
    body_names = [body_set.get(i).getName() for i in range(body_set.getSize())]

    col_labels = list(motion.getColumnLabels())
    motion_np = motion.getMatrix().to_numpy().astype(np.float64)
    times = np.array(motion.getIndependentColumn(), dtype=np.float64)

    # Convert rotation columns from degrees → radians if needed
    in_degrees = False
    try:
        in_degrees = motion.getTableMetaDataAsString("inDegrees").strip().lower() == "yes"
    except Exception:
        pass
    if in_degrees:
        for i, c in enumerate(col_labels):
            try:
                if coord_set.get(c).getMotionType() == 1:  # 1 = rotational
                    motion_np[:, i] *= np.pi / 180.0
            except Exception:
                # Coordinate not in model (extra column) → ignore
                pass

    state = model.initSystem()

    n_frames = motion_np.shape[0]
    transforms_per_body: dict[str, list[list[list[float]]]] = {bn: [] for bn in body_names}

    for n in range(n_frames):
        for c, coord_name in enumerate(col_labels):
            try:
                coord = coord_set.get(coord_name)
            except Exception:
                continue
            try:
                coord.setValue(state, float(motion_np[n, c]), False)
            except Exception:
                pass
        model.realizePosition(state)
        for bn in body_names:
            body = body_set.get(bn)
            H = body.getTransformInGround(state)
            T = H.T()
            R = H.R()
            mat = [
                [R.get(0, 0), R.get(0, 1), R.get(0, 2), T.get(0)],
                [R.get(1, 0), R.get(1, 1), R.get(1, 2), T.get(1)],
                [R.get(2, 0), R.get(2, 1), R.get(2, 2), T.get(2)],
                [0.0, 0.0, 0.0, 1.0],
            ]
            transforms_per_body[bn].append(mat)

    # Per-body mesh local transform as 4x4 (translation + Euler XYZ)
    bodies_out: dict[str, dict] = {}
    for bn in body_names:
        meshes_with_local = []
        for m in body_meshes.get(bn, []):
            R = _eulerXYZ_to_matrix(*m["orientation_xyz"])
            local = np.eye(4)
            local[:3, :3] = R
            local[:3, 3] = np.array(m["translation"], dtype=float)
            meshes_with_local.append({
                "file": m["file"],
                "scale": m["scale"],
                "local_transform": local.tolist(),
            })
        bodies_out[bn] = {
            "meshes": meshes_with_local,
            "world_transforms": transforms_per_body[bn],
        }

    payload = {
        "times": times.tolist(),
        "n_frames": n_frames,
        "bodies": bodies_out,
    }
    Path(out_json).write_text(json.dumps(payload))
    print(f"wrote {n_frames} frames × {len(body_names)} bodies → {out_json}")


if __name__ == "__main__":
    main()
