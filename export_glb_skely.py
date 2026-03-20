#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export OpenSim .mot file to GLB (binary glTF).

GLB works in all external viewers (Three.js, Unity, Unreal, web) and uses
quaternions natively, completely avoiding the Euler angle wrapping issue
that causes 180-degree snaps in FBX viewers.
"""

import os
import sys
from math import radians

# Parse arguments after '--'
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser(description='Export OpenSim .mot to GLB')
parser.add_argument('--mot', required=True, help='Path to .mot motion file')
parser.add_argument('--output', '-o', required=True, help='Output GLB file path')
parser.add_argument('--trc', default=None, help='Optional TRC file for marker-guided spine rotation')
parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
parser.add_argument('--rig', default='metarig_skely', help='Armature name (default: metarig_skely)')
args = parser.parse_args(argv)

import bpy
from mathutils import Euler, Quaternion, Vector
import numpy as np


def unwrap_angle(angles):
    """Unwrap angle array to avoid discontinuities at +/-180 degrees."""
    unwrapped = angles.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i-1]
        if diff > 180:
            unwrapped[i:] -= 360
        elif diff < -180:
            unwrapped[i:] += 360
    return unwrapped


def preprocess_mot_data(headers, data):
    """Preprocess motion data - unwrap angles and center translation at origin."""
    data = np.array(data)

    rot_idx = headers.index('pelvis_rotation') if 'pelvis_rotation' in headers else None
    if rot_idx is not None:
        rotations = data[:, rot_idx].copy()
        unwrapped = unwrap_angle(rotations)
        data[:, rot_idx] = unwrapped
        print(f"  Pelvis rotation: unwrapped (range: {unwrapped.min():.1f} to {unwrapped.max():.1f})")

    tx_idx = headers.index('pelvis_tx') if 'pelvis_tx' in headers else None
    tz_idx = headers.index('pelvis_tz') if 'pelvis_tz' in headers else None

    if tx_idx is not None:
        tx_init = data[0, tx_idx]
        data[:, tx_idx] -= tx_init
        print(f"  pelvis_tx: centered (was {tx_init:.3f}m, now starts at 0)")

    if tz_idx is not None:
        tz_init = data[0, tz_idx]
        data[:, tz_idx] -= tz_init
        print(f"  pelvis_tz: centered (was {tz_init:.3f}m, now starts at 0)")

    return data.tolist()


def read_mot_file(mot_path):
    """Read OpenSim .mot file, return (headers, data_rows, in_degrees)."""
    with open(mot_path, 'r') as f:
        lines = f.readlines()

    in_degrees = True
    data_start = 0
    for i, line in enumerate(lines):
        if 'inDegrees=yes' in line:
            in_degrees = True
        elif 'inDegrees=no' in line:
            in_degrees = False
        if line.strip().startswith('endheader'):
            data_start = i + 1
            break

    headers = None
    data = []
    for i, line in enumerate(lines[data_start:]):
        parts = line.strip().split('\t')
        if i == 0:
            headers = parts
        else:
            try:
                data.append([float(x) for x in parts])
            except ValueError:
                continue

    return headers, data, in_degrees


def get_val(row, headers, col_name, default=0.0):
    """Get value from data row by column name."""
    if col_name in headers:
        idx = headers.index(col_name)
        if idx < len(row):
            return row[idx]
    return default


def read_trc_file(trc_path):
    """Read TRC file, return (marker_names, data) where data[frame][marker] = Vector(x,y,z)."""
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    # Line 4 (index 3) has marker names: Frame# Time Nose  Neck  ...
    parts = lines[3].strip().split('\t')
    marker_names = [m for m in parts[2:] if m.strip()]
    num_markers = len(marker_names)

    # Data starts at line 7 (index 6)
    data = []
    for line in lines[6:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 2 + num_markers * 3:
            continue
        frame_markers = []
        for j in range(num_markers):
            base = 2 + j * 3
            x = float(parts[base])
            y = float(parts[base + 1])
            z = float(parts[base + 2])
            frame_markers.append(Vector((x, y, z)))
        data.append(frame_markers)

    return marker_names, data


def osim_to_blender_vec(v):
    """Convert OpenSim (X=fwd, Y=up, Z=right) to Blender rig coords (-Z, Y, X)."""
    return Vector((-v.z, v.y, v.x))


def distribute_rotation(total_quat, fractions):
    """Distribute a total rotation across N bones using successive slerp.

    Each bone takes its proportional share of the remaining rotation.
    Returns list of N local quaternions whose composition equals total_quat.
    """
    identity = Quaternion((1, 0, 0, 0))
    result = []
    remaining = total_quat.copy()
    frac_left = 1.0

    for i, f in enumerate(fractions):
        if i == len(fractions) - 1:
            result.append(remaining)
        else:
            share = f / frac_left if frac_left > 1e-6 else 0.0
            share = max(0.0, min(1.0, share))
            bone_rot = identity.slerp(remaining, share)
            result.append(bone_rot)
            remaining = bone_rot.inverted() @ remaining
            frac_left -= f

    return result


def apply_quat_with_continuity(bone, quat):
    """Set bone rotation quaternion with sign continuity tracking."""
    bone_name = bone.name
    if bone_name in _prev_quats:
        prev = _prev_quats[bone_name]
        dot = quat.w * prev.w + quat.x * prev.x + quat.y * prev.y + quat.z * prev.z
        if dot < 0:
            quat = Quaternion((-quat.w, -quat.x, -quat.y, -quat.z))
    _prev_quats[bone_name] = quat.copy()
    bone.rotation_quaternion = quat


# Track previous quaternion per bone to ensure sign continuity
_prev_quats = {}


def set_bone_rotation_quat(bone, x, y, z, euler_order='YZX'):
    """Set bone rotation using quaternion with sign continuity."""
    euler = Euler((x, y, z), euler_order)
    quat = euler.to_quaternion()

    bone_name = bone.name
    if bone_name in _prev_quats:
        prev = _prev_quats[bone_name]
        dot = quat.w * prev.w + quat.x * prev.x + quat.y * prev.y + quat.z * prev.z
        if dot < 0:
            quat = Quaternion((-quat.w, -quat.x, -quat.y, -quat.z))

    _prev_quats[bone_name] = quat.copy()
    bone.rotation_quaternion = quat


def apply_motion(rig_name, mot_path, target_fps=30, trc_path=None):
    """Apply .mot motion data to the skeleton using quaternions.

    If trc_path is provided, TRC marker positions drive the torso chain
    (spine.001-003) instead of distributing MOT joint angles heuristically.
    """
    headers, data, in_degrees = read_mot_file(mot_path)
    print(f"Loaded {mot_path}: {len(data)} frames, {len(headers)} columns, degrees={in_degrees}")

    # Load TRC for marker-guided spine rotation
    trc_names, trc_data = None, None
    trc_idx = {}
    if trc_path:
        trc_names, trc_data = read_trc_file(trc_path)
        for name in ['LHip', 'RHip', 'Neck']:
            if name in trc_names:
                trc_idx[name] = trc_names.index(name)
        if len(trc_idx) == 3:
            print(f"  TRC loaded: {len(trc_data)} frames, {len(trc_names)} markers — torso guided by markers")
        else:
            print(f"  TRC missing required markers (LHip/RHip/Neck), falling back to MOT")
            trc_data = None

    print("Preprocessing motion data...")
    data = preprocess_mot_data(headers, data)

    rig = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = rig

    # Remove ALL actions from the blend file - old actions have fcurves
    # targeting "DEF-spine" etc. which confuses the glTF exporter
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    if rig.animation_data:
        rig.animation_data_clear()
    print(f"  Cleared all {len(bpy.data.actions)} remaining actions")

    bpy.ops.object.mode_set(mode='POSE')

    # Set ALL bones to QUATERNION mode (not just animated ones)
    # This prevents "Multiple rotation mode" warnings in glTF export
    for bone in rig.pose.bones:
        bone.rotation_mode = 'QUATERNION'
    print(f"  Set all {len(rig.pose.bones)} bones to QUATERNION rotation mode")

    if len(data) > 1:
        t_ini = data[0][0]
        t_end = data[-1][0]
        source_fps = (len(data) - 1) / (t_end - t_ini) if t_end > t_ini else 30
        fps_skip = max(1, int(source_fps / target_fps))
        print(f"  Source FPS: {source_fps:.1f}, target: {target_fps}, skip: {fps_skip}")
    else:
        fps_skip = 1

    frame = 0
    for i in range(0, len(data), fps_skip):
        row = data[i]

        pelvis_tx = get_val(row, headers, 'pelvis_tx', 0)
        pelvis_ty = get_val(row, headers, 'pelvis_ty', 0)
        pelvis_tz = get_val(row, headers, 'pelvis_tz', 0)
        pelvis_tilt = get_val(row, headers, 'pelvis_tilt', 0)
        pelvis_list = get_val(row, headers, 'pelvis_list', 0)
        pelvis_rotation = get_val(row, headers, 'pelvis_rotation', 0)

        hip_flexion_r = get_val(row, headers, 'hip_flexion_r', 0)
        hip_adduction_r = get_val(row, headers, 'hip_adduction_r', 0)
        hip_rotation_r = get_val(row, headers, 'hip_rotation_r', 0)
        knee_angle_r = get_val(row, headers, 'knee_angle_r', 0)
        ankle_angle_r = get_val(row, headers, 'ankle_angle_r', 0)

        hip_flexion_l = get_val(row, headers, 'hip_flexion_l', 0)
        hip_adduction_l = get_val(row, headers, 'hip_adduction_l', 0)
        hip_rotation_l = get_val(row, headers, 'hip_rotation_l', 0)
        knee_angle_l = get_val(row, headers, 'knee_angle_l', 0)
        ankle_angle_l = get_val(row, headers, 'ankle_angle_l', 0)

        lumbar_ext = get_val(row, headers, 'lumbar_extension',
                    get_val(row, headers, 'L5_S1_Flex_Ext', 0))
        lumbar_bend = get_val(row, headers, 'lumbar_bending',
                     get_val(row, headers, 'L5_S1_Lat_Bending', 0))
        lumbar_rot = get_val(row, headers, 'lumbar_rotation',
                    get_val(row, headers, 'L5_S1_axial_rotation', 0))

        neck_ext = get_val(row, headers, 'thorax_extension',
                  get_val(row, headers, 'neck_flexion', 0))
        neck_bend = get_val(row, headers, 'thorax_bending',
                   get_val(row, headers, 'neck_bending', 0))
        neck_rot = get_val(row, headers, 'thorax_rotation',
                  get_val(row, headers, 'neck_rotation', 0))

        wrist_flex_r = get_val(row, headers, 'wrist_flex_r', 0)
        wrist_dev_r = get_val(row, headers, 'wrist_dev_r', 0)
        wrist_flex_l = get_val(row, headers, 'wrist_flex_l', 0)
        wrist_dev_l = get_val(row, headers, 'wrist_dev_l', 0)

        arm_flex_r = get_val(row, headers, 'arm_flex_r', 0)
        arm_add_r = get_val(row, headers, 'arm_add_r', 0)
        arm_rot_r = get_val(row, headers, 'arm_rot_r', 0)
        elbow_flex_r = get_val(row, headers, 'elbow_flex_r', 0)
        pro_sup_r = get_val(row, headers, 'pro_sup_r', 0)

        arm_flex_l = get_val(row, headers, 'arm_flex_l', 0)
        arm_add_l = get_val(row, headers, 'arm_add_l', 0)
        arm_rot_l = get_val(row, headers, 'arm_rot_l', 0)
        elbow_flex_l = get_val(row, headers, 'elbow_flex_l', 0)
        pro_sup_l = get_val(row, headers, 'pro_sup_l', 0)

        if in_degrees:
            pelvis_tilt = radians(pelvis_tilt)
            pelvis_list = radians(pelvis_list)
            pelvis_rotation = radians(pelvis_rotation)
            hip_flexion_r = radians(hip_flexion_r)
            hip_adduction_r = radians(hip_adduction_r)
            hip_rotation_r = radians(hip_rotation_r)
            knee_angle_r = radians(knee_angle_r)
            ankle_angle_r = radians(ankle_angle_r)
            hip_flexion_l = radians(hip_flexion_l)
            hip_adduction_l = radians(hip_adduction_l)
            hip_rotation_l = radians(hip_rotation_l)
            knee_angle_l = radians(knee_angle_l)
            ankle_angle_l = radians(ankle_angle_l)
            lumbar_ext = radians(lumbar_ext)
            lumbar_bend = radians(lumbar_bend)
            lumbar_rot = radians(lumbar_rot)
            neck_ext = radians(neck_ext)
            neck_bend = radians(neck_bend)
            neck_rot = radians(neck_rot)
            wrist_flex_r = radians(wrist_flex_r)
            wrist_dev_r = radians(wrist_dev_r)
            wrist_flex_l = radians(wrist_flex_l)
            wrist_dev_l = radians(wrist_dev_l)
            arm_flex_r = radians(arm_flex_r)
            arm_add_r = radians(arm_add_r)
            arm_rot_r = radians(arm_rot_r)
            elbow_flex_r = radians(elbow_flex_r)
            pro_sup_r = radians(pro_sup_r)
            arm_flex_l = radians(arm_flex_l)
            arm_add_l = radians(arm_add_l)
            arm_rot_l = radians(arm_rot_l)
            elbow_flex_l = radians(elbow_flex_l)
            pro_sup_l = radians(pro_sup_l)

        # === PELVIS (spine bone) ===
        rig.pose.bones['spine'].location.x = -pelvis_tz
        rig.pose.bones['spine'].location.y = pelvis_ty
        rig.pose.bones['spine'].location.z = pelvis_tx
        rig.pose.bones['spine'].keyframe_insert(data_path="location", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['spine'],
                               -pelvis_tilt, pelvis_rotation, pelvis_list, 'YZX')
        rig.pose.bones['spine'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === RIGHT LEG ===
        set_bone_rotation_quat(rig.pose.bones['thigh.R'],
                               -hip_flexion_r, -hip_rotation_r, -hip_adduction_r, 'YZX')
        rig.pose.bones['thigh.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['shin.R'], knee_angle_r, 0, 0, 'YZX')
        rig.pose.bones['shin.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['foot.R'], -ankle_angle_r, 0, 0, 'YZX')
        rig.pose.bones['foot.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LEFT LEG ===
        set_bone_rotation_quat(rig.pose.bones['thigh.L'],
                               -hip_flexion_l, hip_rotation_l, hip_adduction_l, 'YZX')
        rig.pose.bones['thigh.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['shin.L'], knee_angle_l, 0, 0, 'YZX')
        rig.pose.bones['shin.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['foot.L'], -ankle_angle_l, 0, 0, 'YZX')
        rig.pose.bones['foot.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LOWER LUMBAR (spine.001) — 40% L5_S1 ===
        set_bone_rotation_quat(rig.pose.bones['spine.001'],
                               -lumbar_ext * 0.4, lumbar_rot * 0.4, lumbar_bend * 0.4, 'YZX')
        rig.pose.bones['spine.001'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === UPPER LUMBAR (spine.002) — 60% L5_S1 ===
        set_bone_rotation_quat(rig.pose.bones['spine.002'],
                               -lumbar_ext * 0.6, lumbar_rot * 0.6, lumbar_bend * 0.6, 'YZX')
        rig.pose.bones['spine.002'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === CHEST (spine.003) — identity, keyframed for bake safety ===
        rig.pose.bones['spine.003'].rotation_quaternion = Quaternion((1, 0, 0, 0))
        rig.pose.bones['spine.003'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # Clamp neck to anatomical limits (IK often dumps residual into neck)
        # Values are always in radians at this point (converted above)
        neck_ext  = max(radians(-30), min(radians(30), neck_ext))
        neck_bend = max(radians(-25), min(radians(25), neck_bend))
        neck_rot  = max(radians(-35), min(radians(35), neck_rot))

        # === LOWER NECK (spine.004) — 50% neck ===
        set_bone_rotation_quat(rig.pose.bones['spine.004'],
                               -neck_ext * 0.5, neck_rot * 0.5, neck_bend * 0.5, 'YZX')
        rig.pose.bones['spine.004'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === UPPER NECK (spine.005) — 50% neck ===
        set_bone_rotation_quat(rig.pose.bones['spine.005'],
                               -neck_ext * 0.5, neck_rot * 0.5, neck_bend * 0.5, 'YZX')
        rig.pose.bones['spine.005'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === HEAD (spine.006) — identity, locks to neck chain ===
        rig.pose.bones['spine.006'].rotation_quaternion = Quaternion((1, 0, 0, 0))
        rig.pose.bones['spine.006'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === RIGHT ARM ===
        set_bone_rotation_quat(rig.pose.bones['upper_arm.R'],
                               -arm_flex_r, -arm_rot_r, -arm_add_r, 'YZX')
        rig.pose.bones['upper_arm.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['forearm.R'],
                               -elbow_flex_r, -pro_sup_r, 0, 'YZX')
        rig.pose.bones['forearm.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LEFT ARM ===
        set_bone_rotation_quat(rig.pose.bones['upper_arm.L'],
                               -arm_flex_l, arm_rot_l, arm_add_l, 'YZX')
        rig.pose.bones['upper_arm.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['forearm.L'],
                               -elbow_flex_l, pro_sup_l, 0, 'YZX')
        rig.pose.bones['forearm.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === RIGHT HAND (wrist) ===
        set_bone_rotation_quat(rig.pose.bones['hand.R'],
                               -wrist_flex_r, 0, -wrist_dev_r, 'YZX')
        rig.pose.bones['hand.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LEFT HAND (wrist) ===
        set_bone_rotation_quat(rig.pose.bones['hand.L'],
                               -wrist_flex_l, 0, wrist_dev_l, 'YZX')
        rig.pose.bones['hand.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        frame += 1
        if frame % 200 == 0:
            print(f"  Frame {frame}...")

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frame - 1
    bpy.context.scene.render.fps = target_fps

    print(f"  Applied {frame} frames using QUATERNION rotations")
    return frame


def cleanup_scene(rig_name):
    """Remove objects not related to the skeleton export."""
    keep_names = set()
    rig = bpy.data.objects.get(rig_name)
    if rig:
        keep_names.add(rig.name)
        for obj in bpy.data.objects:
            if obj.parent == rig and obj.type == 'MESH':
                keep_names.add(obj.name)

    to_remove = []
    for obj in bpy.data.objects:
        if obj.name not in keep_names:
            to_remove.append(obj)

    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    print(f"Scene cleaned: kept {len(keep_names)} objects")


def setup_camera():
    """Add a static camera framing the character."""
    cam_data = bpy.data.cameras.new('ExportCamera')
    cam_obj = bpy.data.objects.new('ExportCamera', cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = Vector((0.5, 1.0, 3.0))
    direction = Vector((0.0, 1.0, 0.0)) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    cam_data.lens = 50
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0

    bpy.context.scene.camera = cam_obj
    print(f"Camera added at {tuple(cam_obj.location)}")


def bake_to_deform_bones(rig_name, frame_start, frame_end):
    """Bake visual transforms from control bones to DEF-bones for glTF export.

    The rigify rig has control bones (spine, thigh.R, etc.) that we keyframe,
    and deform bones (DEF-spine, DEF-thigh.R, etc.) that drive the mesh via
    Copy Transform constraints. The glTF exporter only exports deform bones,
    so we need to bake the evaluated constraint results onto them.
    """
    rig = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='POSE')

    # Set all DEF-bones to QUATERNION mode and select them
    bpy.ops.pose.select_all(action='DESELECT')
    def_count = 0
    for bone in rig.pose.bones:
        if bone.name.startswith('DEF-'):
            bone.rotation_mode = 'QUATERNION'
            bone.bone.select = True
            def_count += 1

    print(f"  Baking visual transforms to {def_count} DEF-bones (frames {frame_start}-{frame_end})...")

    # Bake with visual keying - evaluates constraints at each frame
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        bake_types={'POSE'},
    )

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  Baked and cleared constraints on DEF-bones")


def export_glb(output_path):
    """Export scene to GLB (binary glTF) for universal viewer compatibility.

    glTF uses quaternions natively - no Euler angle wrapping issues.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in ['ARMATURE', 'MESH', 'CAMERA']:
            obj.select_set(True)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=True,
        export_animations=True,
        export_animation_mode='ACTIONS',
        export_bake_animation=True,
        export_anim_single_armature=True,
        export_cameras=True,
        export_apply=False,
    )
    print(f"  GLB exported: {output_path}")


def main():
    print(f"Motion file: {args.mot}")
    print(f"Output: {args.output}")
    print(f"Rig: {args.rig}")

    if args.rig not in bpy.data.objects:
        print(f"ERROR: Armature '{args.rig}' not found!")
        print(f"Available armatures: {[o.name for o in bpy.data.objects if o.type == 'ARMATURE']}")
        return

    cleanup_scene(args.rig)
    setup_camera()
    apply_motion(args.rig, args.mot, target_fps=args.fps, trc_path=args.trc)

    # Export GLB (quaternion native, no Euler angle wrapping issues)
    glb_path = args.output
    if not glb_path.endswith('.glb'):
        glb_path = os.path.splitext(glb_path)[0] + '.glb'
    print(f"\nExporting GLB...")
    export_glb(glb_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
