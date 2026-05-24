"""Retarget SAM3D anatomical keypoints onto a rigged MakeHuman avatar GLB.

Approach: "look-at" retargeting from OpenSim marker positions (TRC / kpts array).
For each animated avatar bone we know two anatomical markers that define its
parent and its child end-points. The bone is rotated so that its bind-pose
direction matches the per-frame anatomical direction (parent_marker → child_marker).
Local quaternions are reconstructed by walking the chain.

Pros:
  - Bypasses MHR ↔ SMPL-X axis-angle conversion headaches.
  - Works on any avatar provided we have its bind world matrices, regardless of
    the rest pose (A-pose, T-pose, anything sensible).
Cons:
  - Loses the axial roll along the bone direction (e.g. wrist pronation).
    Acceptable for anonymised exercise playback.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pygltflib
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# MakeHuman 'Default' rig — only the bones we actually drive
# Each entry: (parent_marker, child_marker)
#   parent_marker == ":midhip" sentinel = midpoint(LHJC, RHJC)
#   parent_marker == ":midacr" sentinel = midpoint(LACR, RACR)
# Marker names follow the TRC produced by SAM3D OpenSim exporter.
# ---------------------------------------------------------------------------

# bone: (parent_marker, child_marker, aux_lateral_marker, aux_medial_marker)
# When aux_lateral / aux_medial are non-None, we build a full 3-axis frame to
# capture all 3 DOF (including axial rotation around the bone). Otherwise we
# fall back to a 2-DOF swing-only rotation.
MAKEHUMAN_BONE_TARGETS: dict[str, tuple[str, str, str | None, str | None]] = {
    # Spine chain (2 DOF — no per-segment lateral markers)
    "spine05":      (":midhip",   "c_spine0",   None,         None),
    "spine04":      ("c_spine0",  "c_spine1",   None,         None),
    "spine03":      ("c_spine1",  "c_spine2",   None,         None),
    "spine02":      ("c_spine2",  "c_spine3",   None,         None),
    "spine01":      ("c_spine3",  "C7",         None,         None),
    # Neck / head (2 DOF)
    # neck01 disabled — see note in previous version.
    "neck02":       ("c_neck",    "c_head",     None,         None),
    "head":         ("c_head",    "HTOP",       None,         None),
    # Left arm
    # clavicle.L disabled — C7 marker too noisy.
    # 3 DOF for upperarm only. lowerarm stays at 2 DOF because its bone roll in
    # the MakeHuman A-pose has local X pointing roughly toward (-X, -Y), which
    # doesn't match the anatomical lateral direction we get from the LFAradius/
    # LFAulna markers — using the axial reference there produces a ~180° twist.
    "upperarm01.L": ("LACR",      "LEJC",       "LLEL",       "LMEL"),
    "lowerarm01.L": ("LEJC",      "LWrist_hand", None,        None),
    # Left hand (2 DOF, only fingers with real markers in body-only TRC).
    # Middle (metacarpal2) and ring (metacarpal3) are left at bind because the
    # TRC has no LMiddle/LRing markers — using LIndexTip/LPinky as proxies made
    # them splay weirdly. Will revisit when hand inference adds those markers.
    "finger1-1.L":  ("LWrist_hand", "LThumb",     None, None),
    "metacarpal1.L":("LWrist_hand", "LIndex",     None, None),
    "metacarpal4.L":("LWrist_hand", "LPinky",     None, None),
    # Right arm
    "upperarm01.R": ("RACR",      "REJC",       "RLEL",       "RMEL"),
    "lowerarm01.R": ("REJC",      "RWrist_hand", None,        None),
    # Right hand
    "finger1-1.R":  ("RWrist_hand", "RThumb",     None, None),
    "metacarpal1.R":("RWrist_hand", "RIndex",     None, None),
    "metacarpal4.R":("RWrist_hand", "RPinky",     None, None),
    # Left leg (3 DOF for upperleg + lowerleg, 2 DOF for foot)
    "upperleg01.L": ("LHJC",      "LKJC",       "LLFC",       "LMFC"),
    "lowerleg01.L": ("LKJC",      "LAJC",       "LLMAL",      "LMMAL"),
    "foot.L":       ("LAJC",      "LTOE",       None,         None),
    # Right leg
    "upperleg01.R": ("RHJC",      "RKJC",       "RLFC",       "RMFC"),
    "lowerleg01.R": ("RKJC",      "RAJC",       "RLMAL",      "RMMAL"),
    "foot.R":       ("RAJC",      "RTOE",       None,         None),
}

# Root pelvis: we drive its world translation from midhip and its world rotation
# from the hip line + spine direction. Handled separately from MAKEHUMAN_BONE_TARGETS.
ROOT_BONE_NAME = "root"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AvatarRig:
    gltf: pygltflib.GLTF2
    binary_blob: bytes
    skin_index: int
    joint_node_indices: list[int]                # skin.joints
    joint_names: list[str]                       # name per skin-joint
    joint_to_parent: dict[int, int]              # local joint idx → local joint idx (-1 = root)
    bind_world: np.ndarray                       # (J, 4, 4) bind in world (= inv(IBM))
    bind_local_t: np.ndarray                     # (J, 3) per node local translation (rest)
    bind_local_q: np.ndarray                     # (J, 4) per node local quaternion (x,y,z,w)
    name_to_local_idx: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GLB loading
# ---------------------------------------------------------------------------

def _quat_xyzw_from_node(node: pygltflib.Node) -> np.ndarray:
    if node.rotation is None:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.asarray(node.rotation, dtype=np.float64)


def _t_from_node(node: pygltflib.Node) -> np.ndarray:
    if node.translation is None:
        return np.zeros(3)
    return np.asarray(node.translation, dtype=np.float64)


def load_avatar_glb(path: str | Path) -> AvatarRig:
    gltf = pygltflib.GLTF2().load(str(path))
    blob = gltf.binary_blob()
    skin = gltf.skins[0]

    # Inverse bind matrices → bind world matrices
    acc = gltf.accessors[skin.inverseBindMatrices]
    bv = gltf.bufferViews[acc.bufferView]
    raw = blob[bv.byteOffset:bv.byteOffset + bv.byteLength]
    ibm = np.frombuffer(raw, dtype=np.float32).reshape(acc.count, 4, 4).transpose(0, 2, 1)
    bind_world = np.linalg.inv(ibm).astype(np.float64)

    joint_node_indices = list(skin.joints)
    joint_names = [gltf.nodes[ni].name for ni in joint_node_indices]
    name_to_local_idx = {n: i for i, n in enumerate(joint_names)}

    # Build parent table (local joint idx → local joint idx)
    node_to_local = {ni: i for i, ni in enumerate(joint_node_indices)}
    parent_node_of_joint: dict[int, int] = {}
    for pi, node in enumerate(gltf.nodes):
        if node.children:
            for c in node.children:
                if c in node_to_local:
                    parent_node_of_joint[c] = pi
    joint_to_parent: dict[int, int] = {}
    for ji, ni in enumerate(joint_node_indices):
        pn = parent_node_of_joint.get(ni, -1)
        joint_to_parent[ji] = node_to_local.get(pn, -1)

    # Local TR at bind
    J = len(joint_node_indices)
    bind_local_t = np.zeros((J, 3))
    bind_local_q = np.zeros((J, 4))
    for ji, ni in enumerate(joint_node_indices):
        bind_local_t[ji] = _t_from_node(gltf.nodes[ni])
        bind_local_q[ji] = _quat_xyzw_from_node(gltf.nodes[ni])

    return AvatarRig(
        gltf=gltf,
        binary_blob=blob,
        skin_index=0,
        joint_node_indices=joint_node_indices,
        joint_names=joint_names,
        joint_to_parent=joint_to_parent,
        bind_world=bind_world,
        bind_local_t=bind_local_t,
        bind_local_q=bind_local_q,
        name_to_local_idx=name_to_local_idx,
    )


# ---------------------------------------------------------------------------
# TRC loading
# ---------------------------------------------------------------------------

def load_trc(path: str | Path) -> tuple[np.ndarray, list[str], float]:
    """Return (positions[T, M, 3] in METRES, marker_names, fps)."""
    path = Path(path)
    lines = path.read_text().splitlines()
    rate = float(lines[2].split("\t")[0])
    header = lines[3].split("\t")
    marker_names = [h.strip() for h in header[2:] if h.strip()]
    data_rows = []
    for line in lines[5:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        vals = [float(p) for p in parts[2:] if p.strip()]
        data_rows.append(vals)
    arr = np.array(data_rows, dtype=np.float64)
    T = arr.shape[0]
    M = len(marker_names)
    assert arr.shape[1] == M * 3, f"TRC mismatch: {arr.shape[1]} vs {M*3}"
    positions = arr.reshape(T, M, 3) / 1000.0   # mm → m
    return positions, marker_names, rate


def _marker_pos(name: str, frame_pos: np.ndarray, name_to_idx: dict[str, int],
                left: str = "LHJC", right: str = "RHJC") -> np.ndarray:
    if name == ":midhip":
        return 0.5 * (frame_pos[name_to_idx["LHJC"]] + frame_pos[name_to_idx["RHJC"]])
    if name == ":midacr":
        return 0.5 * (frame_pos[name_to_idx["LACR"]] + frame_pos[name_to_idx["RACR"]])
    return frame_pos[name_to_idx[name]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) rotating unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if d < -0.999999:
        # 180° around any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    s = np.sqrt(2.0 * (1.0 + d))
    inv_s = 1.0 / s
    return np.array([axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, 0.5 * s])


def _trs_to_matrix(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    M[:3, 3] = t
    return M


def _matrix_to_quat_xyzw(M: np.ndarray) -> np.ndarray:
    return R.from_matrix(M[:3, :3]).as_quat()


# ---------------------------------------------------------------------------
# Retargeting core
# ---------------------------------------------------------------------------

@dataclass
class RetargetResult:
    fps: float
    n_frames: int
    # per-frame, per-joint local TR
    root_translation: np.ndarray    # (T, 3) — applied to root bone node
    local_quat: np.ndarray          # (T, J, 4) xyzw quaternions (only retargeted bones differ from bind)


def _bind_world_rot_of(rig: AvatarRig, joint_local_idx: int) -> np.ndarray:
    return rig.bind_world[joint_local_idx, :3, :3].copy()


def retarget_from_trc(
    rig: AvatarRig,
    trc_positions: np.ndarray,            # (T, M, 3) metres
    marker_names: list[str],
    bone_targets: dict[str, tuple[str, str]] = MAKEHUMAN_BONE_TARGETS,
) -> RetargetResult:
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    T = trc_positions.shape[0]
    J = len(rig.joint_names)

    # Initialise local quats to bind rest values for every frame
    local_quat = np.tile(rig.bind_local_q[None, :, :], (T, 1, 1))
    root_translation = np.zeros((T, 3))

    # Precompute, for each driven bone:
    #   - local idx of bone
    #   - bind world rotation
    #   - "bind direction" = (bind_world_pos(child) - bind_world_pos(parent)) normalised,
    #     where child is the next bone in MAKEHUMAN_BONE_TARGETS (we use the
    #     bone's child-end inferred from its own bind world position vs its
    #     own anchor; simpler: use bone's bind-pose head→tail vector).
    # We define bone's bind direction as: bind_world(child_node) - bind_world(this_node).
    # If the bone has no skin child, fallback: use bind_world tail axis = local Y.
    name_to_local = rig.name_to_local_idx

    # Build a quick child lookup: for each joint, pick its first child within the skin
    children_of: dict[int, list[int]] = {ji: [] for ji in range(J)}
    for ji, parent in rig.joint_to_parent.items():
        if parent >= 0:
            children_of[parent].append(ji)

    def bind_direction(bone_name: str) -> np.ndarray:
        """Direction of a bone in avatar world, computed by walking PAST twist
        bones (upperleg02, lowerleg02, upperarm02 — short helper bones inserted
        by MakeHuman that share the same axis but whose head→tail vector points
        in a misleading direction because they're very short).
        """
        ji = name_to_local[bone_name]
        head = rig.bind_world[ji, :3, 3]
        # Walk down through twist children until we hit a "real" anatomical bone.
        # A MakeHuman twist bone has a name ending in "02" with the same prefix.
        current = ji
        for _ in range(4):  # safety bound
            cs = children_of.get(current, [])
            if not cs:
                break
            # Pick the longest child as a heuristic for "the main chain"
            best = max(cs, key=lambda c: np.linalg.norm(rig.bind_world[c, :3, 3] - head))
            best_name = rig.joint_names[best]
            # Stop here if this looks like an anatomical bone (not a twist segment)
            if "02" not in best_name:
                tail = rig.bind_world[best, :3, 3]
                break
            # Otherwise descend through the twist bone
            current = best
        else:
            tail = rig.bind_world[current, :3, 3]
        v = tail - head
        n = np.linalg.norm(v)
        if n < 1e-9:
            return rig.bind_world[ji, :3, 1]
        return v / n

    # bone_meta entries: (joint_idx, bind_dir, p_marker, c_marker, aux_lat, aux_med)
    bone_meta: list[tuple[int, np.ndarray, str, str, str | None, str | None]] = []
    for bone, target_def in bone_targets.items():
        if bone not in name_to_local:
            continue
        pmark, cmark, aux_lat, aux_med = target_def
        ji = name_to_local[bone]
        dir_bind = bind_direction(bone)
        bone_meta.append((ji, dir_bind, pmark, cmark, aux_lat, aux_med))

    # For each frame, walk top-down: compute target world rotation for each driven
    # bone, then convert to local via parent's target world rotation.
    # We need the entire chain's target world rotations even for bones we don't drive,
    # because the avatar has many bones we leave at bind (they keep their bind local
    # rotation, but their world rotation depends on driven ancestors).
    # Strategy: compute target world rotations bone-by-bone in topological order.

    # Build topological order (parents before children)
    order: list[int] = []
    visited = [False] * J
    def dfs(ji: int):
        if visited[ji]:
            return
        p = rig.joint_to_parent.get(ji, -1)
        if p >= 0 and not visited[p]:
            dfs(p)
        visited[ji] = True
        order.append(ji)
    for ji in range(J):
        dfs(ji)

    driven = {entry[0]: entry for entry in bone_meta}  # ji → full meta tuple

    # Pelvis target world rotation: derived from hip line + spine direction.
    root_idx = name_to_local.get(ROOT_BONE_NAME, -1)

    # Reference midhip from frame 0 — we drive the root with the per-frame delta,
    # so that the avatar starts at its natural bind position (feet on the floor)
    # and follows the subject's relative motion (squat depth, walking, etc.).
    midhip_ref = _marker_pos(":midhip", trc_positions[0], name_to_idx)
    root_bind_trans = rig.bind_local_t[root_idx] if root_idx >= 0 else np.zeros(3)

    # === BIND-RELATIVE RETARGETING ===
    # We anchor everything to frame 0: at t=0, the avatar stays exactly in its bind
    # pose (no twist of any kind). At t>0 we apply the delta rotation that the
    # subject's body underwent between frame 0 and frame t. This sidesteps the
    # convention mismatch between OpenSim TRC axes and the Blender/glTF Y-up
    # avatar bind orientation: we never compare absolute frames, only deltas.

    def _pelvis_frame(frame_pos: np.ndarray) -> np.ndarray:
        lh = _marker_pos("LHJC", frame_pos, name_to_idx)
        rh = _marker_pos("RHJC", frame_pos, name_to_idx)
        midhip = _marker_pos(":midhip", frame_pos, name_to_idx)
        mid_acr = _marker_pos(":midacr", frame_pos, name_to_idx)
        x_axis = lh - rh; x_axis /= (np.linalg.norm(x_axis) + 1e-12)
        up_vec = mid_acr - midhip; up_vec /= (np.linalg.norm(up_vec) + 1e-12)
        z_axis = np.cross(x_axis, up_vec); z_axis /= (np.linalg.norm(z_axis) + 1e-12)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    pelvis_frame_ref = _pelvis_frame(trc_positions[0])
    pelvis_frame_ref_inv = pelvis_frame_ref.T  # rotation matrix → inverse = transpose
    bind_world_root_rot = rig.bind_world[root_idx, :3, :3] if root_idx >= 0 else np.eye(3)

    # === True world-to-world alignment (Y-only) ===
    # Both TRC and avatar use Y-up. We want align to flip X/Z if subject and
    # avatar face opposite directions, but NEVER pick up the subject's lean at
    # frame 0 (otherwise we lose absolute pelvis orientation: an already-tilted
    # subject would map to "body identity" and the avatar would stay upright).
    # So we extract only the Y-rotation component of pelvis_frame_ref by
    # projecting the subject's frame-0 facing direction onto the XZ plane.
    _fwd_subj_0 = pelvis_frame_ref[:, 2].copy()
    _fwd_subj_0[1] = 0.0
    _n_fwd = np.linalg.norm(_fwd_subj_0)
    if _n_fwd > 1e-6:
        _fwd_subj_0 /= _n_fwd
        _fwd_av = np.array([0.0, 0.0, 1.0])  # avatar faces +Z at bind
        cos_a = float(np.dot(_fwd_subj_0, _fwd_av))
        sin_a = float(_fwd_subj_0[2] * _fwd_av[0] - _fwd_subj_0[0] * _fwd_av[2])
        align = np.array([[cos_a, 0, sin_a],
                          [    0, 1,     0],
                          [-sin_a, 0, cos_a]])
    else:
        align = np.eye(3)

    def _into_avatar(R_subj: np.ndarray) -> np.ndarray:
        return align @ R_subj @ align.T

    def _vec_into_avatar(v_subj: np.ndarray) -> np.ndarray:
        return align @ v_subj

    # Per-driven-bone bind direction in avatar world.
    bone_bind_dir_avatar: dict[int, np.ndarray] = {}
    for entry in bone_meta:
        ji, dir_bind = entry[0], entry[1]
        bone_bind_dir_avatar[ji] = dir_bind  # already in avatar world via inv(IBM)

    def _make_frame(main_v: np.ndarray, hint_v: np.ndarray) -> np.ndarray:
        """3x3 orthonormal frame: col 0 = main axis, col 1 = `hint_v` projected
        perpendicular to main and normalised, col 2 = right-hand cross."""
        main_v = main_v / (np.linalg.norm(main_v) + 1e-12)
        perp = hint_v - np.dot(hint_v, main_v) * main_v
        n = np.linalg.norm(perp)
        if n < 1e-9:
            # Degenerate: hint parallel to main. Pick an arbitrary perpendicular.
            fallback = np.array([1.0, 0.0, 0.0]) if abs(main_v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp = fallback - np.dot(fallback, main_v) * main_v
            perp /= np.linalg.norm(perp)
        else:
            perp /= n
        third = np.cross(main_v, perp)
        return np.column_stack([main_v, perp, third])

    for t in range(T):
        frame_pos = trc_positions[t]

        # --- Root pelvis translation. Aligned subject world → avatar world
        # (true Y-up mapping now that align is fixed). Vertical motion stays
        # vertical because align is just pelvis_frame_refᵀ which leaves Y alone.
        midhip = _marker_pos(":midhip", frame_pos, name_to_idx)
        delta_midhip = _vec_into_avatar(midhip - midhip_ref)
        root_translation[t] = root_bind_trans + delta_midhip if root_idx >= 0 else delta_midhip

        # --- Pelvis rotation: ABSOLUTE. We express the subject's body axes at
        # frame t in avatar world via the Y-only alignment, then multiply by the
        # avatar's bone-local convention (bind_world_root_rot). This way the
        # avatar's pelvis follows the subject's absolute orientation at every
        # frame — including frame 0. Critical for exercises that start in a
        # non-standard pose (already leaning/squatting).
        try:
            pelvis_frame_t = _pelvis_frame(frame_pos)
            subject_body_in_avatar = align @ pelvis_frame_t
            target_root_world_rot = subject_body_in_avatar @ bind_world_root_rot
        except KeyError:
            target_root_world_rot = bind_world_root_rot

        # Storage of target world rotations per joint for this frame
        target_world_rot = np.tile(np.eye(3), (J, 1, 1))

        for ji in order:
            parent = rig.joint_to_parent.get(ji, -1)
            parent_world = target_world_rot[parent] if parent >= 0 else np.eye(3)
            if ji == root_idx:
                tw = target_root_world_rot
            elif ji in driven and ji in bone_bind_dir_avatar:
                _ji, _dir_bind, pmark, cmark, aux_lat, aux_med = driven[ji]
                try:
                    p_pos = _marker_pos(pmark, frame_pos, name_to_idx)
                    c_pos = _marker_pos(cmark, frame_pos, name_to_idx)
                    dir_t_subj = c_pos - p_pos
                    n = np.linalg.norm(dir_t_subj)
                    if n < 1e-6:
                        tw = parent_world @ R.from_quat(rig.bind_local_q[ji]).as_matrix()
                    else:
                        # ABSOLUTE alignment. Main axis = bone direction.
                        main_av = _vec_into_avatar(dir_t_subj) / n
                        bind_main = bone_bind_dir_avatar[ji]

                        # If axial markers are provided, compute a full 3-axis
                        # frame for both bind and target → captures the rotation
                        # around the bone (e.g. hip rotation, forearm pronation).
                        # Otherwise fall back to 2-DOF swing-only rotation.
                        has_axial = (
                            aux_lat is not None and aux_med is not None
                            and aux_lat in name_to_idx and aux_med in name_to_idx
                        )
                        if has_axial:
                            try:
                                lat_pos = _marker_pos(aux_lat, frame_pos, name_to_idx)
                                med_pos = _marker_pos(aux_med, frame_pos, name_to_idx)
                                aux_subj = lat_pos - med_pos  # lateral → medial
                                aux_av = _vec_into_avatar(aux_subj)
                                # In the avatar's bind, "lateral → medial" of a
                                # LEFT bone (lateral side outside at +X, medial
                                # toward centre) actually means `lat - med`
                                # itself points outward = +X. For RIGHT bones
                                # it's mirrored = −X.
                                if ".L" in rig.joint_names[ji]:
                                    bind_aux = np.array([1.0, 0.0, 0.0])
                                elif ".R" in rig.joint_names[ji]:
                                    bind_aux = np.array([-1.0, 0.0, 0.0])
                                else:
                                    bind_aux = None
                                if bind_aux is not None and np.linalg.norm(aux_av) > 1e-6:
                                    target_frame = _make_frame(main_av, aux_av)
                                    bind_frame = _make_frame(bind_main, bind_aux)
                                    R_delta = target_frame @ bind_frame.T
                                else:
                                    q_delta = _quat_from_two_vectors(bind_main, main_av)
                                    R_delta = R.from_quat(q_delta).as_matrix()
                            except KeyError:
                                q_delta = _quat_from_two_vectors(bind_main, main_av)
                                R_delta = R.from_quat(q_delta).as_matrix()
                        else:
                            q_delta = _quat_from_two_vectors(bind_main, main_av)
                            R_delta = R.from_quat(q_delta).as_matrix()

                        tw = R_delta @ _bind_world_rot_of(rig, ji)
                except KeyError:
                    tw = parent_world @ R.from_quat(rig.bind_local_q[ji]).as_matrix()
            else:
                # Leave at bind local — just propagate parent world
                tw = parent_world @ R.from_quat(rig.bind_local_q[ji]).as_matrix()
            target_world_rot[ji] = tw

            # Convert to local rotation
            local_rot_mat = parent_world.T @ tw
            local_quat[t, ji] = R.from_matrix(local_rot_mat).as_quat()

    return RetargetResult(
        fps=30.0,
        n_frames=T,
        root_translation=root_translation,
        local_quat=local_quat,
    )


# ---------------------------------------------------------------------------
# GLB animation export
# ---------------------------------------------------------------------------

def _pad_to_4bytes(data: bytes) -> bytes:
    pad = (-len(data)) % 4
    return data + b"\x00" * pad


def export_animated_glb(
    rig: AvatarRig,
    result: RetargetResult,
    out_path: str | Path,
    animation_name: str = "exercise",
) -> Path:
    """Append an animation channel set + apply root translation, write a new GLB."""
    gltf = rig.gltf
    blob_in = bytes(rig.binary_blob)
    T = result.n_frames
    fps = result.fps

    # Build new binary buffer: original bytes + animation data
    new_data_chunks: list[bytes] = []
    base = len(blob_in)

    def add_accessor(arr: np.ndarray, comp_type: int, accessor_type: str,
                     mins: list | None = None, maxs: list | None = None) -> int:
        nonlocal base
        # Ensure float32 contiguous
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        bytes_data = arr.tobytes()
        pad = (-len(bytes_data)) % 4
        new_data_chunks.append(bytes_data + b"\x00" * pad)
        bv = pygltflib.BufferView(
            buffer=0,
            byteOffset=base,
            byteLength=len(bytes_data),
        )
        gltf.bufferViews.append(bv)
        bv_idx = len(gltf.bufferViews) - 1
        base += len(bytes_data) + pad

        acc = pygltflib.Accessor(
            bufferView=bv_idx,
            byteOffset=0,
            componentType=comp_type,
            count=arr.shape[0],
            type=accessor_type,
        )
        if mins is not None:
            acc.min = mins
        if maxs is not None:
            acc.max = maxs
        gltf.accessors.append(acc)
        return len(gltf.accessors) - 1

    # Time accessor (shared across channels)
    times = (np.arange(T) / fps).astype(np.float32)
    time_acc = add_accessor(
        times, pygltflib.FLOAT, pygltflib.SCALAR,
        mins=[float(times[0])], maxs=[float(times[-1])],
    )

    # Build channels + samplers
    samplers: list[pygltflib.AnimationSampler] = []
    channels: list[pygltflib.AnimationChannel] = []

    skin = gltf.skins[rig.skin_index]
    joint_node_indices = rig.joint_node_indices
    J = len(joint_node_indices)

    # Driven bones: those whose local_quat differs from bind (within tolerance)
    bind_q = rig.bind_local_q
    diff = np.abs(result.local_quat - bind_q[None, :, :]).max(axis=(0, 2))
    driven_joints = np.where(diff > 1e-4)[0]

    for ji in driven_joints:
        node_idx = joint_node_indices[ji]
        quats = result.local_quat[:, ji, :].astype(np.float32)  # (T, 4) xyzw
        q_acc = add_accessor(quats, pygltflib.FLOAT, pygltflib.VEC4)
        sampler = pygltflib.AnimationSampler(input=time_acc, output=q_acc, interpolation="LINEAR")
        samplers.append(sampler)
        s_idx = len(samplers) - 1
        ch = pygltflib.AnimationChannel(
            sampler=s_idx,
            target=pygltflib.AnimationChannelTarget(node=node_idx, path="rotation"),
        )
        channels.append(ch)

    # Root translation channel
    root_local_idx = rig.name_to_local_idx.get(ROOT_BONE_NAME, -1)
    if root_local_idx >= 0:
        # Convert world-positioned root translation into local: account for the root's
        # own bind local translation if any (typically zero/identity).
        root_node_idx = joint_node_indices[root_local_idx]
        root_t = result.root_translation.astype(np.float32)
        t_acc = add_accessor(root_t, pygltflib.FLOAT, pygltflib.VEC3)
        sampler = pygltflib.AnimationSampler(input=time_acc, output=t_acc, interpolation="LINEAR")
        samplers.append(sampler)
        s_idx = len(samplers) - 1
        ch = pygltflib.AnimationChannel(
            sampler=s_idx,
            target=pygltflib.AnimationChannelTarget(node=root_node_idx, path="translation"),
        )
        channels.append(ch)

    anim = pygltflib.Animation(
        name=animation_name,
        samplers=samplers,
        channels=channels,
    )
    gltf.animations.append(anim)

    # Extend buffer
    new_blob = blob_in + b"".join(new_data_chunks)
    # Update buffer 0 byteLength
    gltf.buffers[0].byteLength = len(new_blob)
    gltf.set_binary_blob(new_blob)

    out_path = Path(out_path)
    gltf.save_binary(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Convenience one-shot
# ---------------------------------------------------------------------------

def generate_avatar_from_trc(
    trc_path: str | Path,
    avatar_glb_path: str | Path,
    out_path: str | Path,
) -> Path:
    rig = load_avatar_glb(avatar_glb_path)
    positions, marker_names, fps = load_trc(trc_path)
    result = retarget_from_trc(rig, positions, marker_names)
    result.fps = fps
    return export_animated_glb(rig, result, out_path)
