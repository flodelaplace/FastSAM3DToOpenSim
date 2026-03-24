"""
OpenSim Scale Tool + IK runner.

Calls the opensim conda environment to:
  1. Scale the generic model to the subject's proportions (Scale Tool).
  2. Run Inverse Kinematics on the scaled model.

Both steps use the opensim Python bindings available in a separate conda env,
invoked here as a subprocess.
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
    # Head / face
    "Nose":         0.4,
    "LEye":         0.3,
    "REye":         0.3,
    "LEar":         0.8,
    "REar":         0.8,
    # Torso
    "Neck":         1.0,
    "LShoulder":    2.0,
    "RShoulder":    2.0,
    # Arms
    "LElbow":       1.0,
    "RElbow":       1.0,
    "LOlecranon":   0.5,
    "ROlecranon":   0.5,
    "LCubitalFossa": 0.5,
    "RCubitalFossa": 0.5,
    "LWrist":       1.0,
    "RWrist":       1.0,
    # Lower body
    "LHip":         2.0,
    "RHip":         2.0,
    "LKnee":        2.0,
    "RKnee":        2.0,
    "LAnkle":       2.0,
    "RAnkle":       2.0,
    # Feet (now in model via Coco133 positions on calcn/toes bodies)
    "LBigToe":      1.5,
    "LSmallToe":    1.0,
    "LHeel":        1.5,
    "RBigToe":      1.5,
    "RSmallToe":    1.0,
    "RHeel":        1.5,
    # Index + Pinky tips only: span the full palm width, sufficient to constrain
    # both wrist_flex and wrist_dev. No MCPs needed — no finger DOFs in model.
    "RIndex":       0.5,   "LIndex":       0.5,
    "RPinky":       0.5,   "LPinky":       0.5,
    # Spine joints from MHR 127-joint armature — now active with the
    # Pose2Sim Wholebody model (explicit lumbar5–1 + torso + head bodies)
    # c_spine0: on sacrum body (MHR idx 34 is at sacral base level)
    "c_spine0":     0.5,   # sacrum body (~sacral base)
    "c_spine1":     0.5,   # lumbar3 body (L3–L4 joint)
    "c_spine2":     0.5,   # torso body, lower thoracic
    "c_spine3":     0.5,   # torso body, upper thoracic/cerviocothoracic
    "c_neck":       0.7,   # torso body, cervical
    "c_head":       0.6,   # head body
}

# ---------------------------------------------------------------------------
# Scale Tool measurement definitions
# Each entry: (name, [(markerA, markerB), ...], [body1, body2, ...], axes_str)
# Markers must exist in both the model MarkerSet and the TRC.
# Axes: which axes of the body to scale (e.g. "X Y Z" = uniform, "Y" = height only).
# ---------------------------------------------------------------------------
_SCALE_MEASUREMENTS = [
    # Pelvis + sacrum: width from inter-hip distance
    ("pelvis",       [("LHip",      "RHip")],        ["pelvis", "sacrum"],                                                         "X Y Z"),
    # Trunk height: sacrum-to-neck (Y only — width handled separately)
    ("torso_height", [("c_spine0",  "Neck")],         ["torso", "Abdomen", "lumbar1", "lumbar2", "lumbar3", "lumbar4", "lumbar5"],  "Y"),
    # Head: use same trunk-height scale factor (best proxy for overall body scale).
    # Direct Neck-c_head pair is unreliable (landmark mismatch gives 0.5×).
    ("head",         [("c_spine0",  "Neck")],         ["head"],                                                                     "X Y Z"),
    # Trunk width: shoulder-to-shoulder (X,Z — height handled separately)
    ("torso_width",  [("LShoulder", "RShoulder")],    ["torso", "Abdomen", "lumbar1", "lumbar2", "lumbar3", "lumbar4", "lumbar5"],  "X Z"),
    # Right lower limb
    ("femur_r",      [("RHip",      "RKnee")],        ["femur_r", "patella_r"],                                                     "X Y Z"),
    ("tibia_r",      [("RKnee",     "RAnkle")],       ["tibia_r"],                                                                  "X Y Z"),
    ("foot_r",       [("RHeel",     "RBigToe")],      ["talus_r", "calcn_r", "toes_r"],                                             "X Y Z"),
    # Right upper limb
    ("humerus_r",    [("RShoulder", "RElbow")],       ["humerus_r"],                                                                "X Y Z"),
    ("forearm_r",    [("RElbow",    "RWrist")],       ["ulna_r", "radius_r"],                                                       "X Y Z"),
    # hand_r: no scale measurement. Template positions are set to actual generic model
    # fingertip geometry (index_distal_rvs/little_distal_rvs at 0.85× model scale).
    # Cross-body pairs (RWrist→RIndex) are unreliable: body-mode finger predictions are
    # noisy across walking frames (std >160mm) → ScaleTool gets garbage K values.
    # Left lower limb
    ("femur_l",      [("LHip",      "LKnee")],        ["femur_l", "patella_l"],                                                     "X Y Z"),
    ("tibia_l",      [("LKnee",     "LAnkle")],       ["tibia_l"],                                                                  "X Y Z"),
    ("foot_l",       [("LHeel",     "LBigToe")],      ["talus_l", "calcn_l", "toes_l"],                                             "X Y Z"),
    # Left upper limb
    ("humerus_l",    [("LShoulder", "LElbow")],       ["humerus_l"],                                                                "X Y Z"),
    ("forearm_l",    [("LElbow",    "LWrist")],       ["ulna_l", "radius_l"],                                                       "X Y Z"),
    # Note: head excluded from direct measurement — Neck-c_head landmark mismatch gives 0.5×.
    # head is instead included in torso_height (above) to scale uniformly with the body.
]


def _write_scale_setup_xml(
    model_path: str,
    trc_path: str,
    output_model_path: str,
    scale_set_path: str,
    mass: float,
    height_mm: float,
    t_start: float,
    t_end: float,
    trc_marker_names: list[str],
    xml_path: str,
) -> None:
    # ScaleTool resolves marker_file and output_model_file relative to the XML
    # file's directory. Use relative paths; keep model_file absolute (loaded
    # from the assets directory, not the output directory).
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    trc_rel          = os.path.relpath(os.path.abspath(trc_path),           xml_dir)
    out_model_rel    = os.path.relpath(os.path.abspath(output_model_path),  xml_dir)
    scale_set_rel    = os.path.relpath(os.path.abspath(scale_set_path),     xml_dir)

    trc_set = set(trc_marker_names)

    meas_xml_parts = []
    for name, pairs, bodies, axes in _SCALE_MEASUREMENTS:
        # Skip if any marker in this measurement is missing from TRC
        if not all(a in trc_set and b in trc_set for a, b in pairs):
            continue

        pair_xml = "\n".join(
            f'\t\t\t\t\t\t\t<MarkerPair>\n'
            f'\t\t\t\t\t\t\t\t<markers> {a} {b} </markers>\n'
            f'\t\t\t\t\t\t\t</MarkerPair>'
            for a, b in pairs
        )
        body_xml = "\n".join(
            f'\t\t\t\t\t\t\t<BodyScale name="{body}">\n'
            f'\t\t\t\t\t\t\t\t<axes> {axes} </axes>\n'
            f'\t\t\t\t\t\t\t</BodyScale>'
            for body in bodies
        )
        meas_xml_parts.append(
            f'\t\t\t\t\t\t<Measurement name="{name}">\n'
            f'\t\t\t\t\t\t\t<apply>true</apply>\n'
            f'\t\t\t\t\t\t\t<MarkerPairSet>\n'
            f'\t\t\t\t\t\t\t\t<objects>\n'
            f'{pair_xml}\n'
            f'\t\t\t\t\t\t\t\t</objects>\n'
            f'\t\t\t\t\t\t\t</MarkerPairSet>\n'
            f'\t\t\t\t\t\t\t<BodyScaleSet>\n'
            f'\t\t\t\t\t\t\t\t<objects>\n'
            f'{body_xml}\n'
            f'\t\t\t\t\t\t\t\t</objects>\n'
            f'\t\t\t\t\t\t\t</BodyScaleSet>\n'
            f'\t\t\t\t\t\t</Measurement>'
        )

    meas_str = "\n".join(meas_xml_parts)

    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
\t<ScaleTool name="subject">
\t\t<mass>{mass:.2f}</mass>
\t\t<height>{height_mm:.1f}</height>
\t\t<age>-1</age>
\t\t<GenericModelMaker>
\t\t\t<model_file>{model_path}</model_file>
\t\t</GenericModelMaker>
\t\t<ModelScaler>
\t\t\t<apply>true</apply>
\t\t\t<scaling_order> measurements </scaling_order>
\t\t\t<MeasurementSet>
\t\t\t\t<objects>
{meas_str}
\t\t\t\t</objects>
\t\t\t</MeasurementSet>
\t\t\t<marker_file>{trc_rel}</marker_file>
\t\t\t<time_range>{t_start:.6f} {t_end:.6f}</time_range>
\t\t\t<output_model_file>{out_model_rel}</output_model_file>
\t\t\t<output_scale_file>{scale_set_rel}</output_scale_file>
\t\t\t<preserve_mass_distribution>true</preserve_mass_distribution>
\t\t</ModelScaler>
\t\t<MarkerPlacer>
\t\t\t<apply>false</apply>
\t\t\t<output_model_file></output_model_file>
\t\t</MarkerPlacer>
\t</ScaleTool>
</OpenSimDocument>
"""
    Path(xml_path).write_text(xml)


# ---------------------------------------------------------------------------
# Inline Scale Tool script executed inside the opensim env
# ---------------------------------------------------------------------------
_SCALE_SCRIPT = """
import sys, json, os, xml.etree.ElementTree as ET
setup_xml        = sys.argv[1]
result_json      = sys.argv[2]
scaled_model_path = sys.argv[3]
scale_set_path   = sys.argv[4]

try:
    import opensim
    opensim.Logger.setLevelString('error')

    # Snapshot marker positions BEFORE ScaleTool — these are the template (pre-scale)
    # positions we use as the authoritative source for marker placement.
    _pre_model = opensim.Model(scaled_model_path)
    _pre_model.initSystem()
    _pre_ms = _pre_model.getMarkerSet()
    _pre_pos = {}
    for _i in range(_pre_ms.getSize()):
        _m = _pre_ms.get(_i)
        _loc = _m.get_location()
        _pre_pos[_m.getName()] = (_loc[0], _loc[1], _loc[2])

    tool = opensim.ScaleTool(setup_xml)
    tool.run()

    # --- Fix marker local positions ---
    # OpenSim's ScaleTool applies non-uniform scaling to finger markers: MCPs are
    # scaled correctly by the body scale factor, but some fingertip markers (Pinky,
    # Middle, Ring) get over-scaled by up to 14% beyond the body scale. This pushes
    # those marker balls beyond the finger mesh geometry, causing a visual "double
    # distance" effect in the OpenSim viewer.
    #
    # Fix: always recompute every marker's local position as:
    #     new_pos = template_pos × body_scale_factor
    # This gives perfectly uniform scaling regardless of what OpenSim's ScaleTool
    # wrote internally. Reading from _pre_pos (template) avoids any double-scaling.
    scale_factors = {}   # body_name -> [sx, sy, sz]
    if os.path.isfile(scale_set_path):
        tree = ET.parse(scale_set_path)
        for sc in tree.findall('.//Scale'):
            seg = sc.find('segment')
            val = sc.find('scales')
            if seg is not None and val is not None:
                body_name = seg.text.strip()
                vals = [float(v) for v in val.text.split()]
                scale_factors[body_name] = vals

    if scale_factors:
        # Bodies without a Scale Tool measurement (e.g. hand_r/l) may still be
        # scaled by OpenSim via parent-body inheritance.  Force those to 1.0×
        # so their template positions (set from actual mesh geometry) are preserved.
        model = opensim.Model(scaled_model_path)
        model.initSystem()
        ms = model.getMarkerSet()
        for i in range(ms.getSize()):
            m = ms.get(i)
            name = m.getName()
            body_name = m.getParentFrameName().split('/')[-1]
            if name in _pre_pos:
                sf = scale_factors.get(body_name, [1.0, 1.0, 1.0])
                pre = _pre_pos[name]
                m.set_location(opensim.Vec3(pre[0]*sf[0], pre[1]*sf[1], pre[2]*sf[2]))
        model.printToXML(scaled_model_path)

    json.dump({"ok": True}, open(result_json, "w"))
except Exception as e:
    json.dump({"ok": False, "error": str(e)}, open(result_json, "w"))
    sys.exit(1)
"""


def run_scale_tool(
    model_path: str,
    trc_path: str,
    scaled_model_path: str,
    subject_mass: float = 70.0,
    subject_height: float = 1.75,
) -> bool:
    """
    Scale the generic OpenSim model to the subject's proportions using the TRC.

    Uses all available frames for robust average segment-length estimation.
    Returns True on success, False if the opensim env is unavailable or fails.
    """
    opensim_python = _find_opensim_python()
    if opensim_python is None:
        return False

    t_start, t_end = _get_trc_time_range(trc_path)
    trc_marker_names = _read_trc_marker_names(trc_path)
    output_dir = str(Path(scaled_model_path).parent.resolve())

    with tempfile.TemporaryDirectory() as tmp:
        xml_path       = os.path.join(output_dir, "_scale_setup.xml")
        scale_set_path = os.path.join(output_dir, "_scale_factors.xml")
        result_json    = os.path.join(tmp, "result.json")
        script_path    = os.path.join(tmp, "run_scale.py")

        _write_scale_setup_xml(
            model_path=os.path.abspath(model_path),
            trc_path=os.path.abspath(trc_path),
            output_model_path=os.path.abspath(scaled_model_path),
            scale_set_path=os.path.abspath(scale_set_path),
            mass=subject_mass,
            height_mm=subject_height * 1000.0,
            t_start=t_start,
            t_end=t_end,
            trc_marker_names=trc_marker_names,
            xml_path=xml_path,
        )
        Path(script_path).write_text(_SCALE_SCRIPT)

        result = subprocess.run(
            [opensim_python, script_path, xml_path, result_json,
             os.path.abspath(scaled_model_path), os.path.abspath(scale_set_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  [Scale] OpenSim Scale Tool failed:\n{result.stderr[-500:]}")
            return False

        if os.path.exists(result_json):
            r = json.load(open(result_json))
            if not r.get("ok"):
                print(f"  [Scale] OpenSim Scale Tool error: {r.get('error')}")
                return False

    for f in (xml_path, scale_set_path):
        try:
            os.remove(f)
        except OSError:
            pass

    return os.path.isfile(scaled_model_path)


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
