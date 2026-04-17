"""Compute OpenSim anatomical-bone transforms + load mesh files for GLB export.

Bridge between the opensim conda env (used for IK) and the main env (which
writes the GLB). Uses a subprocess just like opensim_ik_runner.py does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Reuse the same opensim Python lookup as the IK runner
from .opensim_ik_runner import _find_opensim_python


def _default_geometry_dir() -> str | None:
    """Best-effort lookup of a Pose2Sim Geometry folder shipped with the repo."""
    candidates = []
    # Pose2Sim package shipped with the main env (most likely path)
    try:
        import Pose2Sim  # noqa: F401
        p2s_root = Path(sys.modules["Pose2Sim"].__file__).parent
        candidates.append(p2s_root / "OpenSim_Setup" / "Geometry")
    except Exception:
        pass
    # Env override
    if env := os.environ.get("OPENSIM_GEOMETRY_PATH"):
        candidates.append(Path(env))
    # Local repo override (assets/Geometry if present)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "assets" / "Geometry")
    for c in candidates:
        if c and c.is_dir():
            return str(c)
    return None


def compute_body_transforms(osim_path: str | Path, mot_path: str | Path) -> dict | None:
    """Run the opensim subprocess to compute per-frame body transforms.

    Returns a dict (parsed JSON) with structure:
        {"times": [...], "n_frames": N,
         "bodies": {body_name: {"meshes": [...], "world_transforms": [[4x4], ...]}}}
    or None if the opensim env was not found / the subprocess failed.
    """
    osim_path = Path(osim_path).resolve()
    mot_path = Path(mot_path).resolve()
    if not osim_path.is_file() or not mot_path.is_file():
        return None
    py = _find_opensim_python()
    if py is None:
        print("[opensim_anatomical] opensim conda env not found, skipping.")
        return None
    helper = Path(__file__).with_name("_opensim_compute_body_transforms.py")
    out_json = osim_path.parent / (osim_path.stem + "_body_transforms.json")
    cmd = [py, str(helper), str(osim_path), str(mot_path), str(out_json)]
    # Force UTF-8 decoding and provide locale to child to avoid ASCII crashes
    # on OpenSim's accented warning messages.
    child_env = {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"}
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[opensim_anatomical] subprocess failed:\n  stdout: {exc.stdout}\n  stderr: {exc.stderr}")
        return None
    try:
        return json.loads(out_json.read_text())
    except Exception as exc:
        print(f"[opensim_anatomical] failed to parse {out_json}: {exc}")
        return None


def load_geometry_meshes(
    bodies: dict,
    geometry_dir: str | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load .stl/.vtp meshes referenced by each body.

    Returns {body_name: [(verts_Nx3, faces_Mx3), ...]} where verts/faces are
    already in the body-local frame (i.e. mesh.local_transform + mesh.scale applied).

    Tries .stl first (faster), falls back to .vtp via vtk if needed.
    Bodies whose meshes aren't found in `geometry_dir` are skipped silently.
    """
    if geometry_dir is None:
        geometry_dir = _default_geometry_dir()
    if geometry_dir is None or not Path(geometry_dir).is_dir():
        print("[opensim_anatomical] no Geometry/ folder found, skipping anatomical bones.")
        return {}

    import trimesh

    geom_dir = Path(geometry_dir)
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    missing: list[str] = []

    # Individual vertebrae meshes (cerv*, thoracic*_s) have vertex positions
    # designed for per-vertebra body frames, not the consolidated torso body.
    # When a composite mesh (hat_ribs_scap / hat_spine) is present on the same
    # body, it provides the correct combined visualization — skip all individual
    # vertebrae.  This matches OpenSim GUI / Pose2Sim_Blender behaviour where
    # the composite mesh covers the mispositioned vertebrae.
    _VERTEBRA_PREFIXES = ("cerv", "thoracic")

    for body_name, body_data in bodies.items():
        mesh_list = body_data.get("meshes", [])
        mesh_files = {Path(m["file"]).stem for m in mesh_list}
        has_composite = "hat_ribs_scap" in mesh_files or "hat_spine" in mesh_files
        meshes_for_body = []
        for m in mesh_list:
            f = m["file"]
            stem = Path(f).stem
            if has_composite and stem.startswith(_VERTEBRA_PREFIXES):
                continue
            # Prefer .stl (loadable by trimesh alone, no vtk needed),
            # then .ply, then fall back to .vtp (requires vtk).
            candidates = [
                geom_dir / f"{stem}.stl",
                geom_dir / f"{stem}.ply",
                geom_dir / f"{stem}.obj",
                geom_dir / f,            # as written in the .osim (usually .vtp)
                geom_dir / f"{stem}.vtp",
            ]
            mesh_path = next((c for c in candidates if c.is_file()), None)
            if mesh_path is None:
                missing.append(f)
                continue

            try:
                if mesh_path.suffix.lower() == ".vtp":
                    mesh_obj = _load_vtp(str(mesh_path))
                else:
                    mesh_obj = trimesh.load(mesh_path, force="mesh")
            except Exception as exc:
                print(f"[opensim_anatomical] failed to load {mesh_path}: {exc}")
                continue
            if mesh_obj is None or len(mesh_obj.vertices) == 0:
                continue

            verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
            faces = np.asarray(mesh_obj.faces, dtype=np.uint32)

            # Apply per-mesh scale + local transform (mesh frame → body frame)
            scale = np.asarray(m.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
            verts = verts * scale[None, :]
            local = np.asarray(m.get("local_transform", np.eye(4).tolist()), dtype=np.float32)
            R_loc = local[:3, :3]
            t_loc = local[:3, 3]
            verts = verts @ R_loc.T + t_loc[None, :]

            meshes_for_body.append((verts, faces))
        if meshes_for_body:
            out[body_name] = meshes_for_body

    if missing:
        print(f"[opensim_anatomical] {len(missing)} mesh files missing in {geom_dir} (showing first 5): {missing[:5]}")
    return out


def _load_vtp(path: str):
    """Lazy VTK fallback for .vtp files."""
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        print("[opensim_anatomical] vtk not installed, cannot read .vtp. Run: pip install vtk")
        return None
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    n_cells = poly.GetNumberOfCells()
    faces = []
    for i in range(n_cells):
        cell = poly.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 3:
            faces.append([cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)])
        elif n == 4:
            ids = [cell.GetPointId(j) for j in range(4)]
            faces.append([ids[0], ids[1], ids[2]])
            faces.append([ids[0], ids[2], ids[3]])
    import trimesh
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int32), process=False)
