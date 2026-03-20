"""
OpenSim IK runner.

Calls the opensim conda environment to run Inverse Kinematics on a TRC file
using the Pose2Sim_Simple body model.  The opensim Python bindings are in a
separate conda env and are invoked here as a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Marker weights for IK – matching Pose2Sim defaults
MARKER_WEIGHTS: dict[str, float] = {
    "Nose":         0.4,
    "LEar":         0.8,
    "REar":         0.8,
    "Neck":         1.0,
    "LShoulder":    2.0,
    "RShoulder":    2.0,
    "LElbow":       1.0,
    "RElbow":       1.0,
    "LWrist":       1.0,
    "RWrist":       1.0,
    "LIndex3":      0.5,
    "RIndex3":      0.5,
    "LMiddleTip":   0.5,
    "RMiddleTip":   0.5,
    "LHip":         2.0,
    "RHip":         2.0,
    "LKnee":        2.0,
    "RKnee":        2.0,
    "LAnkle":       2.0,
    "RAnkle":       2.0,
}

# Absolute path to opensim conda Python (adjust if env is installed elsewhere)
_OPENSIM_PYTHON_CANDIDATES = [
    "/home/linuxaitor/miniconda3/envs/opensim/bin/python",
    "/opt/conda/envs/opensim/bin/python",
]


def _find_opensim_python() -> str | None:
    for p in _OPENSIM_PYTHON_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _read_trc_marker_names(trc_path: str) -> list[str]:
    """Return list of marker names from TRC header (line 4)."""
    with open(trc_path) as f:
        for i, line in enumerate(f):
            if i == 3:  # 0-indexed: line 4
                parts = line.strip().split("\t")
                # Skip Frame# and Time, then every other entry (name, empty, empty)
                names = []
                idx = 2
                while idx < len(parts):
                    name = parts[idx].strip()
                    if name:
                        names.append(name)
                    idx += 3
                return names
    return []


def _write_ik_setup_xml(
    model_path: str,
    trc_path: str,
    mot_path: str,
    time_start: float,
    time_end: float,
    trc_marker_names: list[str],
    xml_path: str,
) -> None:
    trc_set = set(trc_marker_names)
    tasks_xml = []
    for name, weight in MARKER_WEIGHTS.items():
        apply = "true" if name in trc_set else "false"
        tasks_xml.append(
            f'\t\t\t<IKMarkerTask name="{name}">\n'
            f'\t\t\t\t<apply>{apply}</apply>\n'
            f'\t\t\t\t<weight>{weight}</weight>\n'
            f'\t\t\t</IKMarkerTask>'
        )
    tasks_str = "\n".join(tasks_xml)

    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
\t<InverseKinematicsTool name="IK">
\t\t<model_file>{model_path}</model_file>
\t\t<marker_file>{trc_path}</marker_file>
\t\t<output_motion_file>{mot_path}</output_motion_file>
\t\t<time_range>{time_start:.6f} {time_end:.6f}</time_range>
\t\t<accuracy>1e-05</accuracy>
\t\t<constraint_weight>Inf</constraint_weight>
\t\t<IKTaskSet>
\t\t\t<objects>
{tasks_str}
\t\t\t</objects>
\t\t</IKTaskSet>
\t</InverseKinematicsTool>
</OpenSimDocument>
"""
    Path(xml_path).write_text(xml)


# ---------------------------------------------------------------------------
# Inline IK script executed inside the opensim env
# ---------------------------------------------------------------------------
_IK_SCRIPT = """
import sys, json, os
setup_xml = sys.argv[1]
result_json = sys.argv[2]
output_dir  = sys.argv[3]   # cwd is changed here so errors file lands in output_dir

try:
    import opensim
    opensim.Logger.setLevelString('error')
    os.chdir(output_dir)
    tool = opensim.InverseKinematicsTool(setup_xml)
    tool.run()
    json.dump({"ok": True}, open(result_json, "w"))
except Exception as e:
    json.dump({"ok": False, "error": str(e)}, open(result_json, "w"))
    sys.exit(1)
"""


def run_ik(
    model_path: str,
    trc_path: str,
    mot_path: str,
    errors_path: str | None = None,
) -> bool:
    """
    Run OpenSim IK and write the resulting MOT to *mot_path*.

    Returns True on success, False if opensim env is unavailable or IK fails.
    The *errors_path* argument is currently unused (OpenSim writes errors to
    the same directory automatically as <stem>_ik_marker_errors.sto).
    """
    opensim_python = _find_opensim_python()
    if opensim_python is None:
        print("  [IK] opensim conda env not found – skipping IK")
        return False

    # Read time range from TRC
    time_start, time_end = _get_trc_time_range(trc_path)
    trc_marker_names = _read_trc_marker_names(trc_path)

    output_dir = str(Path(mot_path).parent.resolve())

    with tempfile.TemporaryDirectory() as tmp:
        # Write setup XML to the OUTPUT dir so that OpenSim writes the marker
        # errors file alongside it (OpenSim uses setup XML dir, not CWD).
        xml_path    = os.path.join(output_dir, "_ik_setup.xml")
        result_json = os.path.join(tmp, "result.json")
        script_path = os.path.join(tmp, "run_ik.py")

        _write_ik_setup_xml(
            model_path=os.path.abspath(model_path),
            trc_path=os.path.abspath(trc_path),
            mot_path=os.path.abspath(mot_path),
            time_start=time_start,
            time_end=time_end,
            trc_marker_names=trc_marker_names,
            xml_path=xml_path,
        )
        Path(script_path).write_text(_IK_SCRIPT)

        result = subprocess.run(
            [opensim_python, script_path, xml_path, result_json, output_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  [IK] OpenSim IK failed:\n{result.stderr}")
            return False

        if os.path.exists(result_json):
            r = json.load(open(result_json))
            if not r.get("ok"):
                print(f"  [IK] OpenSim IK error: {r.get('error')}")
                return False

    # Clean up setup XML
    try:
        os.remove(xml_path)
    except OSError:
        pass

    # OpenSim writes marker errors to the output dir as "IK_ik_marker_errors.sto"
    # (tool name "IK" + "_ik_marker_errors.sto").  Rename to expected convention.
    output_dir_path = Path(output_dir)  # absolute, set above
    auto_errors = output_dir_path / "IK_ik_marker_errors.sto"
    target_errors = Path(errors_path) if errors_path else (output_dir_path / "_ik_marker_errors.sto")
    if auto_errors.exists():
        auto_errors.rename(target_errors)

    return True


def _get_trc_time_range(trc_path: str) -> tuple[float, float]:
    """Return (start_time, end_time) from a TRC file."""
    times = []
    with open(trc_path) as f:
        for i, line in enumerate(f):
            if i < 6:          # skip header
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[1]))
            except ValueError:
                pass
    if not times:
        return 0.0, 0.0
    return times[0], times[-1]
