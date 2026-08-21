#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fast SAM 3D Body – OpenSim Video Export
========================================
Matches SAM3D-OpenSim output format exactly.  Writes to <output_dir>/:

  markers_<name>_skeleton.mp4     — annotated video with 2D skeleton overlay
  markers_<name>.trc               — 73 markers in mm (Y-up, OpenSim coords)
  markers_<name>_ik.mot            — joint angles from OpenSim IK solver (40 DOF)
  markers_<name>_model.osim        — Pose2Sim Wholebody body model for IK
  markers_<name>_mesh.glb          — full body mesh GLB + skeleton overlay (--no_mesh_glb to skip)
  markers_<name>_anatomical.glb    — OpenSim anatomical bones animated by IK
  inference_meta.json              — video metadata
  video_outputs.json               — per-frame raw 3D keypoints
  processing_report.json           — pipeline summary and timings

Usage:
    conda activate fast_sam_3d_body

    SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 \\
    USE_TRT_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 COMPILE_MODE=reduce-overhead \\
    MHR_NO_CORRECTIVES=1 GPU_HAND_PREP=1 BODY_INTERM_PRED_LAYERS=0,2 \\
    DEBUG_NAN=0 PARALLEL_DECODERS=0 COMPILE_WARMUP_BATCH_SIZES=1 \\
    python demo_video_opensim.py \\
        --video_path ./videos/aitor_garden_walk.mp4 \\
        --detector yolo_pose \\
        --detector_model checkpoints/yolo/yolo11m-pose.engine \\
        --fx 1371

Coordinate system (TRC)
------------------------
  OpenSim Y-up:  X = forward (anterior), Y = up, Z = right (lateral)
  Units: mm (millimetres) — matches SAM3D-OpenSim convention.

Post-processing pipeline
-------------------------
  1. PostProcessor        — missing-frame interpolation, Butterworth 6 Hz low-pass filter
  2. CoordinateTransformer — camera → OpenSim Y-up axes, uniform height scaling
                              (c_head from jcoords[113] as exact top reference,
                               user-provided --person_height as ground truth),
                              pelvis centering, per-frame ground alignment
  3. KeypointConverter    — MHR70 → OpenSim marker names
                             (body + hands + derived PelvisCenter/Thorax
                              + real spine joints c_spine0–3/c_neck/c_head from jcoords)
  4. TRCExporter          — writes .trc in mm
  5. OpenSim IK           — runs InverseKinematicsTool via opensim conda env
                             → produces _ik.mot (40 DOF) and _ik_marker_errors.sto
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.export.opensim_exporter import write_mesh_glb, write_combined_mesh_glb
from sam_3d_body.export.post_processing import PostProcessor
from sam_3d_body.export.coordinate_transform import CoordinateTransformer
from sam_3d_body.export.keypoint_converter import KeypointConverter
from sam_3d_body.export.trc_exporter import TRCExporter
from sam_3d_body.export.avatar_retarget import generate_avatar_from_trc
from sam_3d_body.export.opensim_ik_runner import (
    run_ik, run_scale_tool, run_com_analysis,
    run_per_marker_error_analysis,
)

# Pose2Sim Wholebody model — has explicit lumbar5–lumbar1 spine segments
# so that MHR armature spine joints (c_spine0–3, c_neck, c_head) actually
# drive individual intervertebral DOFs in OpenSim IK.
_POSE2SIM_MODEL_TEMPLATE = os.path.join(parent_dir, "assets", "pose2sim_wholebody_model.osim")



_GLTFPACK = os.environ.get("GLTFPACK_PATH") or shutil.which("gltfpack") or ""

def _fix_morph_weights(data: bytes) -> bytes:
    """Fix gltfpack's illegal quantization of morph weight animation outputs.

    gltfpack converts morph weight outputs to UNSIGNED_BYTE/normalized, which violates
    the glTF spec (weights must be FLOAT). This re-encodes them back to float32.
    """
    import json as _json
    json_len = struct.unpack_from('<I', data, 12)[0]
    gltf = _json.loads(data[20:20+json_len])
    bin_data = bytearray(data[20+json_len+8:])

    anim = gltf.get('animations', [{}])[0]
    weights_channels = [c for c in anim.get('channels', []) if c['target']['path'] == 'weights']
    fixed = False
    for ch in weights_channels:
        sampler = anim['samplers'][ch['sampler']]
        acc = gltf['accessors'][sampler['output']]
        if acc['componentType'] == 5126:   # already float32
            continue
        # Decode from quantized type → float32
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        ct = acc['componentType']
        normalized = acc.get('normalized', False)
        dtype_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16}
        raw = np.frombuffer(bytes(bin_data[offset:offset + count * np.dtype(dtype_map[ct]).itemsize]),
                            dtype=dtype_map[ct])
        if normalized:
            scale = {5120: 1/127, 5121: 1/255, 5122: 1/32767, 5123: 1/65535}[ct]
            values = raw.astype(np.float32) * scale
        else:
            values = raw.astype(np.float32)
        float_bytes = values.tobytes()
        # Append float32 data to end of binary buffer and add a new bufferView
        new_bv_offset = len(bin_data)
        bin_data.extend(float_bytes)
        pad = (4 - len(float_bytes) % 4) % 4
        bin_data.extend(b'\x00' * pad)
        new_bv_idx = len(gltf['bufferViews'])
        gltf['bufferViews'].append({'buffer': 0, 'byteOffset': new_bv_offset, 'byteLength': len(float_bytes)})
        acc['bufferView'] = new_bv_idx
        acc['byteOffset'] = 0
        acc['componentType'] = 5126
        acc.pop('normalized', None)
        fixed = True

    if not fixed:
        return data

    gltf['buffers'][0]['byteLength'] = len(bin_data)
    json_bytes = _json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * pad_j
    bin_bytes = bytes(bin_data)
    pad_b = (4 - len(bin_bytes) % 4) % 4
    bin_bytes += b'\x00' * pad_b
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack('<II', len(bin_bytes),  0x004E4942) + bin_bytes
    header = struct.pack('<III', 0x46546C67, 2, 12 + len(json_chunk) + len(bin_chunk))
    return header + json_chunk + bin_chunk


def _compress_glb(path: str) -> None:
    """Run gltfpack -c on a GLB, fix illegal weight quantization, replace in-place."""
    if not os.path.isfile(_GLTFPACK):
        return
    tmp = path + ".tmp.glb"
    result = subprocess.run(
        [_GLTFPACK, "-c", "-i", path, "-o", tmp],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.isfile(tmp):
        orig_mb = os.path.getsize(path) / 1e6
        fixed = _fix_morph_weights(open(tmp, 'rb').read())
        open(tmp, 'wb').write(fixed)
        comp_mb = os.path.getsize(tmp) / 1e6
        os.replace(tmp, path)
        print(f"  [gltfpack] {orig_mb:.1f} MB → {comp_mb:.1f} MB ({100*comp_mb/orig_mb:.0f}%)")
    else:
        if os.path.isfile(tmp):
            os.remove(tmp)
        print(f"  [gltfpack] compression failed: {result.stderr[-200:]}")


def draw_results_on_frame(img_bgr, outputs, visualizer):
    out = img_bgr.copy()
    for person in outputs:
        if "bbox" in person:
            b = person["bbox"]
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        if "pred_keypoints_2d" in person and visualizer is not None:
            kpts = person["pred_keypoints_2d"]
            kpts_with_score = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=-1)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out_rgb = visualizer.draw_skeleton(out_rgb, kpts_with_score)
            out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out


def _marker_region(name, opensim_body):
    """Regroupe un marqueur en région anatomique (color-coding overlay).
    Porté de Mesh2Sim (demo_pipeline_mono._marker_region) — c'est ce qui rend
    73 marqueurs lisibles d'un coup d'œil."""
    n = (name or "").upper()
    b = opensim_body or ""
    if b == "head" or n in {"HTOP", "NOSE", "LEYE", "REYE", "LEAR", "REAR"}:
        return "head"
    if b in {"torso", "lumbar3", "lumbar5", "pelvis"} or n.startswith(("C_", "C7", "T")) \
            or n in {"RACR", "LACR", "RCLAV", "LCLAV", "RASI", "LASI", "RPSI", "LPSI",
                     "XIPH", "RGTR", "LGTR"}:
        return "trunk"
    if "hand" in b:
        return "hand_r" if b.endswith("_r") else "hand_l"
    if any(k in b for k in ("foot", "calcn", "talus", "toes")):
        return "foot_r" if b.endswith("_r") or n.startswith("R") else "foot_l"
    if n.startswith("R"):
        return "leg_r" if any(k in b for k in ("femur", "tibia", "patella")) else "arm_r"
    if n.startswith("L"):
        return "leg_l" if any(k in b for k in ("femur", "tibia", "patella")) else "arm_l"
    return "other"


def _load_anatomical_markerset():
    """Charge le markerset anatomique Synkro (73 marqueurs virtuels).
    Retourne (names, vertex_indices[np.int], regions, opensim_bodies) ou None.
    Chaque marqueur = 1 index de vertex du mesh MHR (pred_vertices)."""
    import json
    cj = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "assets", "correspondence_sole.json")
    if not os.path.isfile(cj):
        return None
    d = json.load(open(cj))
    markers = d.get("markers", [])
    names, idxs, regions, bodies = [], [], [], []
    for m in markers:
        vids = m.get("mhr_vertices") or []
        if not vids:
            continue
        nm = m.get("name")
        body = m.get("opensim_body") or ""
        names.append(nm)
        idxs.append(int(vids[0]))          # 1er vertex listé (cf. converter)
        bodies.append(body)
        regions.append(_marker_region(nm, body))
    return names, np.asarray(idxs, dtype=np.int64), regions, bodies


def _project_points(P3, cam_t, focal, width, height):
    """Projette des points 3D (repère réseau SAM3D, mètres) → pixels image.
    pred_keypoints_2d = K @ (P3 + cam_t), K = [[f,0,W/2],[0,f,H/2],[0,0,1]]
    (principal point au centre, convention SAM3D). Retourne (N,2) float, NaN
    pour les points derrière la caméra."""
    cx, cy = width / 2.0, height / 2.0
    Xc = P3 + cam_t[None, :]
    Z = Xc[:, 2:3]
    uv = np.full((P3.shape[0], 2), np.nan, dtype=np.float64)
    ok = (Z[:, 0] > 1e-6)
    uv[ok, 0] = focal * Xc[ok, 0] / Z[ok, 0] + cx
    uv[ok, 1] = focal * Xc[ok, 1] / Z[ok, 0] + cy
    return uv


# Squelette anatomique : segments de tracé sur un marqueur représentatif par
# articulation. Les 73 marqueurs anatomiques sont des landmarks de SURFACE
# (RLFC=condyle latéral, XIPH=devant, C7=derrière) → les relier donne un
# squelette incohérent. On trace donc le squelette sur les CENTRES ARTICULAIRES
# du modèle (keypoints MHR body, centrés sur l'articulation) → cohérent + centré.
# Nom JC_* → nom du keypoint mhr70 (projeté depuis pred_keypoints_3d, validé 0px).
_JOINT_CENTERS = [
    ("JC_neck", "neck"),               ("JC_nose", "nose"),
    ("JC_Lshoulder", "left_shoulder"), ("JC_Rshoulder", "right_shoulder"),
    ("JC_Lelbow", "left_elbow"),       ("JC_Relbow", "right_elbow"),
    ("JC_Lwrist", "left_wrist"),       ("JC_Rwrist", "right_wrist"),
    ("JC_Lhip", "left_hip"),           ("JC_Rhip", "right_hip"),
    ("JC_Lknee", "left_knee"),         ("JC_Rknee", "right_knee"),
    ("JC_Lankle", "left_ankle"),       ("JC_Rankle", "right_ankle"),
    ("JC_Lheel", "left_heel"),         ("JC_Rheel", "right_heel"),
    ("JC_Ltoe", "left_big_toe"),       ("JC_Rtoe", "right_big_toe"),
]
# Squelette cinématique cohérent sur les centres articulaires (ligne centrale).
_JOINT_EDGES = [
    ("JC_neck", "JC_nose"),
    ("JC_neck", "JC_Lshoulder"), ("JC_neck", "JC_Rshoulder"),
    ("JC_Lshoulder", "JC_Lelbow"), ("JC_Lelbow", "JC_Lwrist"),
    ("JC_Rshoulder", "JC_Relbow"), ("JC_Relbow", "JC_Rwrist"),
    ("JC_Lshoulder", "JC_Lhip"), ("JC_Rshoulder", "JC_Rhip"),
    ("JC_Lhip", "JC_Rhip"),
    ("JC_Lhip", "JC_Lknee"), ("JC_Lknee", "JC_Lankle"),
    ("JC_Lankle", "JC_Lheel"), ("JC_Lheel", "JC_Ltoe"),
    ("JC_Rhip", "JC_Rknee"), ("JC_Rknee", "JC_Rankle"),
    ("JC_Rankle", "JC_Rheel"), ("JC_Rheel", "JC_Rtoe"),
]

# Chaîne de la COLONNE (centres de vertèbres c_spine*, sur l'axe → cohérent).
# Bas → haut, base reliée au sacrum (RPSI/LPSI). Se raccorde aux épaules via les
# arêtes C7↔RACR/LACR de _ANAT_VOLUME_EDGES.
_SPINE_EDGES = [
    ("c_spine0", "RPSI"), ("c_spine0", "LPSI"),
    ("c_spine0", "c_spine2"), ("c_spine2", "c_spine3"), ("c_spine3", "C7"),
    ("C7", "c_neck"), ("c_neck", "c_head"), ("c_head", "HTOP"),
]

# Bandes de VOLUME entre landmarks anatomiques appariés (médial + latéral au
# même niveau → "barreaux" + arêtes latérale/médiale de chaque segment). Donne
# du volume au membre (wireframe), sans incohérence avant/arrière.
_ANAT_VOLUME_EDGES = [
    # jambe droite : cuisse (GTR→condyles), barreau genou, tibia (condyles→
    # malléoles), barreau cheville, pied.
    ("RGTR", "RLFC"), ("RGTR", "RMFC"), ("RLFC", "RMFC"),
    ("RLFC", "RLMAL"), ("RMFC", "RMMAL"), ("RLMAL", "RMMAL"),
    ("RLMAL", "RCAL"), ("RMMAL", "RCAL"), ("RCAL", "RTOE"),
    ("RTOE", "RMT5"), ("RCAL", "RMT5"),
    # jambe gauche
    ("LGTR", "LLFC"), ("LGTR", "LMFC"), ("LLFC", "LMFC"),
    ("LLFC", "LLMAL"), ("LMFC", "LMMAL"), ("LLMAL", "LMMAL"),
    ("LLMAL", "LCAL"), ("LMMAL", "LCAL"), ("LCAL", "LTOE"),
    ("LTOE", "LMT5"), ("LCAL", "LMT5"),
    # bras droit : bras (ACR→épicondyles), barreau coude, avant-bras, barreau poignet
    ("RACR", "RLEL"), ("RACR", "RMEL"), ("RLEL", "RMEL"),
    ("RLEL", "RFAradius"), ("RMEL", "RFAulna"), ("RFAradius", "RFAulna"),
    # bras gauche
    ("LACR", "LLEL"), ("LACR", "LMEL"), ("LLEL", "LMEL"),
    ("LLEL", "LFAradius"), ("LMEL", "LFAulna"), ("LFAradius", "LFAulna"),
    # tronc + bassin (quad pelvis, côtés du tronc, épaules, cou→tête)
    ("RACR", "LACR"), ("C7", "RACR"), ("C7", "LACR"), ("HTOP", "C7"),
    ("RASI", "LASI"), ("RPSI", "LPSI"), ("RASI", "RPSI"), ("LASI", "LPSI"),
    ("RASI", "RGTR"), ("LASI", "LGTR"), ("RPSI", "RGTR"), ("LPSI", "LGTR"),
    ("RACR", "RASI"), ("LACR", "LASI"),
]


def _project_or_none(P3, cam_t, focal, width, height):
    """Projette [M,3] → liste de [x,y,conf] (conf 0 si hors-champ/derrière cam)."""
    uv = _project_points(np.asarray(P3), np.asarray(cam_t), float(focal),
                         width, height)
    out = []
    for x, y in uv:
        if np.isnan(x) or np.isnan(y):
            out.append([None, None, 0.0])
        else:
            out.append([round(float(x), 2), round(float(y), 2), 1.0])
    return out


def write_points2d_json_3d(all_anat, all_cam_t, all_focal, names, regions,
                           label_set, width, height, fps, out_path,
                           all_kpts_raw=None, all_kpts_2d=None,
                           kpt_name_to_id=None, offset_s=0.0):
    """Écrit points2d.json (coords écran) pour l'overlay interactif front-end.

    DEUX groupes de points, projetés via K @ (P + cam_t) (validé 0px) :
    1. **73 marqueurs anatomiques** (index vertex du mesh) = dots "mocap" colorés
       par région (approche Mesh2Sim). PAS de segments entre eux (landmarks de
       surface → squelette incohérent).
    2. **Centres articulaires** (JC_*, keypoints MHR body centrés) = squelette
       cohérent (`edges`), à tracer par le front en couleurs Synkro (bleu Navy).

    Champs : `regions` (color-coding ; les JC ont la région "joint"),
    `label_markers`, `edges` (sur les JC). offset_s=0. Auto-validation reproj si
    all_kpts_raw/all_kpts_2d fournis.
    """
    import json
    n_anat = len(names)

    # Indices mhr70 des centres articulaires (via kpt_name_to_id).
    jc_names, jc_ids = [], []
    if kpt_name_to_id:
        for jc_name, kpt_name in _JOINT_CENTERS:
            if kpt_name in kpt_name_to_id:
                jc_names.append(jc_name)
                jc_ids.append(kpt_name_to_id[kpt_name])
    jc_ids = np.asarray(jc_ids, dtype=np.int64) if jc_ids else None

    frames = []
    for anat, k3, cam_t, focal in zip(all_anat, all_kpts_raw or [None] * len(all_anat),
                                      all_cam_t, all_focal):
        n_total = n_anat + len(jc_names)
        if cam_t is None or not focal:
            frames.append([[None, None, 0.0] for _ in range(n_total)])
            continue
        # 1. marqueurs anatomiques
        if anat is not None:
            fr = _project_or_none(anat, cam_t, focal, width, height)
        else:
            fr = [[None, None, 0.0] for _ in range(n_anat)]
        # 2. centres articulaires
        if jc_ids is not None and k3 is not None:
            fr += _project_or_none(np.asarray(k3)[jc_ids], cam_t, focal, width, height)
        else:
            fr += [[None, None, 0.0] for _ in range(len(jc_names))]
        frames.append(fr)

    # Auto-validation projection (reprojette les kpts connus).
    reproj_err = None
    if all_kpts_raw is not None and all_kpts_2d is not None:
        errs = []
        for k3, k2, cam_t, focal in zip(all_kpts_raw, all_kpts_2d, all_cam_t, all_focal):
            if k3 is None or k2 is None or cam_t is None or not focal:
                continue
            uv = _project_points(np.asarray(k3), np.asarray(cam_t), float(focal),
                                 width, height)
            d = np.linalg.norm(uv - np.asarray(k2)[:, :2], axis=1)
            errs.extend(d[np.isfinite(d)].tolist())
        if errs:
            reproj_err = float(np.median(errs))

    all_names = list(names) + jc_names
    all_regions = list(regions) + ["joint"] * len(jc_names)
    _nidx = {nm: i for i, nm in enumerate(all_names)}
    # `edges` = squelette central (centres articulaires + chaîne colonne).
    # `edges_volume` = bandes médial/latéral entre landmarks → volume (wireframe).
    edges = [[_nidx[a], _nidx[b]] for a, b in (_JOINT_EDGES + _SPINE_EDGES)
             if a in _nidx and b in _nidx]
    edges_volume = [[_nidx[a], _nidx[b]] for a, b in _ANAT_VOLUME_EDGES
                    if a in _nidx and b in _nidx]

    doc = {
        "schema_version": "1.0.0",
        "modality": "3d",
        "video": {"largeur_px": int(width), "hauteur_px": int(height),
                  "fps": round(float(fps), 6)},
        "sync": {"offset_s": offset_s},
        "noms": all_names,
        "regions": all_regions,
        "label_markers": sorted(label_set) if label_set else [],
        "edges": edges,
        "edges_volume": edges_volume,
        "frames": frames,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f)
    return len(frames), len(all_names), len(edges) + len(edges_volume), reproj_err


# Palette overlay bakée (BGR) — MÊME DA que le pipeline 2D gonio :
#   squelette BLANC · marqueurs Navy pleins uniformes (pas de couleur par
#   région) · Lime en accent (ici les centres articulaires).
# Réf : synkro-gonio2d/scripts/runner.py, _patch_skeleton_uniform_color()
# et _patch_render_cosmetics().
_OVERLAY_NAVY = (75, 24, 23)     # #17184B (bleu Synkro)
_OVERLAY_LIME = (3, 241, 217)    # #D9F103
_OVERLAY_WHITE = (255, 255, 255)


def render_overlay_video(points2d_path, raw_video_path, out_path):
    """Rend l'overlay Synkro (marqueurs colorés par région + squelette bleu Navy
    + volume) depuis points2d.json sur la vidéo BRUTE → overlay baké dans le
    dossier de résultats. MÊME rendu (données + algo) que ce que le front
    dessinera → source unique. Retourne le nb de frames écrites."""
    import json
    doc = json.load(open(points2d_path))
    names = doc["noms"]
    regions = doc.get("regions", ["other"] * len(names))
    label_set = set(doc.get("label_markers", []))
    edges = doc.get("edges", [])
    edges_vol = doc.get("edges_volume", [])
    frames = doc["frames"]
    fps = doc["video"]["fps"]
    cap = cv2.VideoCapture(str(raw_video_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    n = 0
    while cap.isOpened() and n < len(frames):
        ret, img = cap.read()
        if not ret:
            break
        fr = frames[n]
        # 1. bandes de volume : blanc fin (wireframe discret du membre)
        for a, b in edges_vol:
            pa, pb = fr[a], fr[b]
            if pa[0] is None or pb[0] is None or pa[2] == 0 or pb[2] == 0:
                continue
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     _OVERLAY_WHITE, 1, cv2.LINE_AA)
        # 2. squelette central : blanc (DA gonio, thickness 2)
        for a, b in edges:
            pa, pb = fr[a], fr[b]
            if pa[0] is None or pb[0] is None or pa[2] == 0 or pb[2] == 0:
                continue
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     _OVERLAY_WHITE, 2, cv2.LINE_AA)
        # 3. points : marqueurs Navy pleins uniformes ; centres articulaires en
        #    Lime (accent Synkro, comme le point focus du 2D).
        for i, name in enumerate(names):
            p = fr[i]
            if p[0] is None or p[2] == 0:
                continue
            x, y = int(p[0]), int(p[1])
            if not (0 <= x < W and 0 <= y < H):
                continue
            if regions[i] == "joint":
                cv2.circle(img, (x, y), 7, _OVERLAY_LIME, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (x, y), 5, _OVERLAY_NAVY, -1, cv2.LINE_AA)
                if name in label_set:
                    cv2.putText(img, name, (x + 7, y - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _OVERLAY_WHITE, 1,
                                cv2.LINE_AA)
        writer.write(img)
        n += 1
    cap.release()
    writer.release()
    return n


def _centroid_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _lean_angle_at_frame(keypoints, frame_idx):
    """Measure pelvis→thorax lean angle (degrees) on a single frame.

    Positive = forward lean, negative = backward lean.
    """
    kp = keypoints[frame_idx]
    pelvis = (kp[9] + kp[10]) / 2
    thorax = (kp[67] + kp[68]) / 2
    spine_vec = thorax - pelvis
    xz = np.array([spine_vec[0], spine_vec[1]])
    if np.linalg.norm(xz) < 0.01:
        return 0.0
    cos_a = np.dot(xz, [0, 1]) / np.linalg.norm(xz)
    a = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return float(a if spine_vec[0] > 0 else -a)


def _detect_static_window(kpts_opensim: np.ndarray, fps: float,
                          window_sec: float = 0.4):
    """Find the best static-pose window for Scale Tool / MarkerPlacer.

    Scoring: sum of per-keypoint frame-to-frame displacements over a
    rolling *window_sec* window. The window with minimum motion is the
    most static segment. Returns (t_start, t_end, frame_start, frame_end).
    """
    N = kpts_opensim.shape[0]
    window = max(5, int(round(window_sec * fps)))
    if N < window + 2:
        return 0.0, (N - 1) / fps, 0, N - 1

    # Per-frame displacement between consecutive frames, summed over kpts.
    diff = np.diff(kpts_opensim, axis=0)                 # (N-1, 70, 3)
    disp = np.linalg.norm(diff, axis=-1)                 # (N-1, 70)
    motion = np.nansum(disp, axis=-1)                    # (N-1,)

    # Rolling sum over (window - 1) consecutive differences.
    kernel = np.ones(window - 1)
    rolling = np.convolve(motion, kernel, mode="valid")  # len N - window
    best_start = int(np.argmin(rolling))
    f_start = best_start
    f_end   = best_start + window - 1
    return f_start / fps, f_end / fps, f_start, f_end


def _lean_angle_over_range(keypoints, center_frame, half_window=5):
    """Average pelvis→thorax lean angle over a window of frames.

    Averages over [center - half_window, center + half_window], skipping
    frames outside array bounds.  Much more robust than a single frame.
    """
    N = keypoints.shape[0]
    lo = max(0, center_frame - half_window)
    hi = min(N, center_frame + half_window + 1)
    angles = []
    for i in range(lo, hi):
        a = _lean_angle_at_frame(keypoints, i)
        angles.append(a)
    if not angles:
        return 0.0
    return float(np.median(angles))


def main(args, estimator=None, visualizer=None):
    """Traite une video.

    `estimator` / `visualizer` sont injectables pour qu'un worker persistant
    charge les modeles UNE fois (73 s mesurees) au lieu d'a chaque video. En
    ligne de commande ils restent None et le comportement est inchange.
    """
    # Auto-generate timestamped output directory (matches SAM3D-OpenSim convention)
    if args.output_dir is None:
        video_name_raw = os.path.splitext(os.path.basename(args.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output_{timestamp}_{video_name_raw}"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Markerset selection ──────────────────────────────────────────────────
    # --markerset flodelaplace (default) : FlodelaplaceConverter v2 + Model_Flodelaplace_XIPH.osim
    #                                       (73 markers via Mesh2Marker correspondence)
    # --markerset flodelaplace_legacy    : FlodelaplaceConverter v1 + flodelaplace_mocap.osim
    #                                       (64 markers, KPT70+JCOORD127+vertex_idx merge — kept for back-compat)
    # --markerset pose2sim               : original KeypointConverter + pose2sim_wholebody_model.osim
    markerset = getattr(args, "markerset", "pose2sim")
    if markerset == "flodelaplace":
        model_template = os.path.join(parent_dir, "assets", "Model_Flodelaplace_SOLE.osim")
        from sam_3d_body.export.flodelaplace_converter import FlodelaplaceConverter
        florian_converter = FlodelaplaceConverter()
        # Force vertex collection even when mesh GLB is disabled — the
        # converter needs the anatomical vertex positions per frame.
        force_collect_verts = True
    elif markerset == "flodelaplace_legacy":
        model_template = os.path.join(parent_dir, "assets", "flodelaplace_mocap.osim")
        # Legacy converter still lives in git history; import the current
        # module (which is now v2). If you really need the old behavior,
        # check out commit before this migration.
        from sam_3d_body.export.flodelaplace_converter import FlodelaplaceConverter
        florian_converter = FlodelaplaceConverter()
        force_collect_verts = True
    else:
        model_template = _POSE2SIM_MODEL_TEMPLATE
        florian_converter = None
        force_collect_verts = False
    print(f"Markerset: {markerset}  (model template: {os.path.basename(model_template)})")

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    cam_int = None
    if args.fx is not None:
        fx = args.fx
        fy = args.fy if args.fy is not None else fx
        cx = args.cx if args.cx is not None else 0.0
        cy = args.cy if args.cy is not None else 0.0
        _K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                           dtype=torch.float32).unsqueeze(0).cuda()
        cam_int = _K
        print(f"Using fixed intrinsics: fx={fx:.1f} fy={fy:.1f}")

    # ── Model loading ─────────────────────────────────────────────────────────
    if estimator is None:
        print("Loading SAM 3D Body model...")
        t_load = time.time()
        estimator = setup_sam_3d_body(
            detector_name=args.detector,
            detector_model=args.detector_model,
            local_checkpoint_path=args.local_checkpoint,
        )
        print(f"Model loaded in {time.time() - t_load:.1f}s")
    else:
        print("Reusing preloaded SAM 3D Body model (worker persistant)")
    if visualizer is None:
        visualizer = SkeletonVisualizer(line_width=2, radius=5)
        visualizer.set_pose_meta(mhr70_pose_info)

    # Optionally estimate floor tilt from MoGe on frame 0 and skip spine correction
    # when MoGe estimation is active (avoids overcorrection).
    moge_floor_angle = None
    moge_floor_quality = None
    # Flag : True = MoGe a essayé et confirme que le sol est indétectable → skip
    # tout fix. False = MoGe pas encore essayé OU a foiré techniquement → fallback OK.
    moge_all_rejected = False
    # Mise au sol MoGe ACTIVEE PAR DEFAUT, comme dans generate_avatars.py.
    # Ce pipeline-ci exigeait le flag explicite, alors que son aide annonce
    # deja le contraire : en local sans --floor_moge, AUCUNE mise au sol
    # n'etait faite. Le TRC sortait avec le bassin a Y=0 et les pieds 70 cm
    # sous le sol, et tout ce qui en derive suivait — dont les fleches GRF,
    # ancrees sur les pieds du TRC. Invisible sur AWS, ou le lambda ajoute
    # --floor_moge explicitement.
    _floor_moge_on = (getattr(args, "floor_moge", False)
                      or (not args.floor and not args.floor_seated
                          and not getattr(args, "no_floor_moge", False)))
    if _floor_moge_on and not args.no_lean_fix:
        fov_est = getattr(estimator, "fov_estimator", None)
        if fov_est is not None:
            # MULTI-FRAME robust estimation : sample N frames dispersées,
            # filtre par netteté (Laplacien variance), RANSAC → median.
            # Env var MOGE_MULTI_FRAME=0 pour revenir à l'ancien mode single-frame.
            multi_frame = bool(int(os.environ.get("MOGE_MULTI_FRAME", "1")))
            n_samples = int(os.environ.get("MOGE_N_SAMPLES", "8"))
            t_moge = time.time()
            if multi_frame:
                print(f"\nEstimating floor plane from MoGe (multi-frame, N={n_samples})...")
                # Use YOLO detector directly (rapide, ~10x plus léger que
                # process_one_image qui refait tout le pipeline SAM3D). Le sol
                # a pas besoin de précision millimétrique sur le bbox — juste
                # exclure grossièrement la zone sujet.
                _yolo = getattr(estimator, "detector", None)
                def _bbox_fn(frame_rgb):
                    if _yolo is None:
                        return None
                    try:
                        # Passe direct au détecteur YOLO (bypass SAM3D backbone)
                        results = _yolo.predict(frame_rgb, verbose=False)
                        if not results or len(results[0].boxes) == 0:
                            return None
                        # Premier bbox (le plus confiant)
                        b = results[0].boxes.xyxy[0].cpu().numpy()
                        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    except Exception:
                        return None
                try:
                    pitch, roll, quality = CoordinateTransformer.robust_floor_angle_multi_frame(
                        args.video_path,
                        depth_estimator_fn=fov_est.get_depth_points,
                        person_bbox_fn=_bbox_fn,
                        n_samples=n_samples,
                    )
                    moge_floor_angle = (pitch, roll)
                    moge_floor_quality = quality
                    print(f"  MoGe floor tilt: pitch={pitch:+.2f}° roll={roll:+.2f}° "
                          f"[{quality['status']}, {quality['n_used']}/{quality['n_requested']} frames, "
                          f"pitch_std={quality.get('pitch_std_deg', 0):.2f}°] "
                          f"(took {time.time() - t_moge:.2f}s)")
                    # Debug détail par frame
                    n_aberrant = sum(1 for pf in quality.get("per_frame", [])
                                     if pf.get("status") == "rejected_aberrant_plane")
                    n_blurry = sum(1 for pf in quality.get("per_frame", [])
                                   if pf.get("status") == "skipped_blurry")
                    if n_aberrant or n_blurry:
                        print(f"  [floor_moge] rejects: aberrant={n_aberrant} blurry={n_blurry}")
                    if quality.get("status") == "unstable":
                        print(f"  [floor_moge] ⚠ unstable estimation (std > threshold). "
                              f"Consider --floor + --lean_ref_frame N for manual correction.")
                    elif quality.get("status") == "insufficient_samples":
                        # SKIP entièrement le lean fix — mieux que corriger avec un mauvais angle.
                        # moge_all_rejected empêche le fallback single-frame (mêmes bugs).
                        print(f"  [floor_moge] ⚠ MoGe unreliable ({quality['n_used']}/{quality['n_requested']} "
                              f"clean frames, aberrant={n_aberrant}, blurry={n_blurry}). "
                              f"Skipping camera-pitch correction — skeleton kept as-is. "
                              f"Use --lean_ref_frame N to correct manually if needed.")
                        moge_floor_angle = None
                        moge_all_rejected = True
                except Exception as e:
                    print(f"  [floor_moge] multi-frame failed ({e}) — fallback single frame.")
                    multi_frame = False
            if (not multi_frame or moge_floor_angle is None) and not moge_all_rejected:
                # Fallback single-frame (ancien comportement) — uniquement si le
                # multi-frame a foiré techniquement, PAS si MoGe a rejeté toutes
                # les frames (dans ce cas, ne pas insister avec un mauvais angle).
                print("Estimating floor plane from MoGe depth (frame 0)...")
                cap_m = cv2.VideoCapture(args.video_path)
                ret_m, first_frame = cap_m.read()
                cap_m.release()
                if ret_m and first_frame is not None:
                    try:
                        pts, mask = fov_est.get_depth_points(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                        first_bbox = None
                        try:
                            outs = estimator.process_one_image(
                                cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                                hand_box_source=args.hand_box_source,
                                inference_type=args.inference_type,
                                bbox_thr=getattr(args, 'bbox_thr', None),
                                nms_thr=getattr(args, 'nms_thr', None),
                            )
                            if outs and "bbox" in outs[0]:
                                first_bbox = tuple(outs[0]["bbox"])
                        except Exception:
                            first_bbox = None
                        moge_floor_angle = CoordinateTransformer.floor_angle_from_moge_points(
                            pts, mask, person_bbox=first_bbox,
                            orig_hw=(first_frame.shape[0], first_frame.shape[1]),
                        )
                        _p, _r = moge_floor_angle
                        print(f"  MoGe floor tilt (frame 0): pitch={_p:+.2f}° roll={_r:+.2f}° "
                              f"(took {time.time() - t_moge:.2f}s)")
                    except Exception:
                        print("  [floor_moge] MoGe floor estimation failed — skipping.")
        else:
            print("  [floor_moge] No FOV estimator available — skipping.")

    # ── Video I/O ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames > 0:
        total = min(total, args.max_frames)

    frame_step = max(1, round(fps / args.target_fps)) if args.target_fps > 0 else 1
    out_fps    = fps / frame_step

    # ── Intrinsèques caméra : une fois, pas à chaque frame ────────────────────
    # Sans --fx, `cam_int` reste None et l'estimateur relance MoGe sur CHAQUE
    # frame pour ré-estimer une focale qui ne bouge pas (10,4 s mesurées sur une
    # vidéo de 285 frames, soit 18 % de la boucle d'inférence). On échantillonne
    # N frames et on prend la médiane : plus rapide, et plus stable qu'une focale
    # qui fluctue d'une frame à l'autre.
    # Hypothèse : pas de zoom pendant la vidéo. C'est pourquoi c'est opt-in —
    # une prise de vue moto avec zoom violerait cette hypothèse.
    if cam_int is None and getattr(args, "fov_once", 0) > 0:
        _fov = getattr(estimator, "fov_estimator", None)
        if _fov is None:
            print("  [fov_once] pas d'estimateur FOV, ignoré")
        else:
            _n = max(1, min(args.fov_once, max(1, total)))
            _t_fov = time.time()
            _fx_s, _fy_s = [], []
            for _i in np.linspace(0, max(0, total - 1), _n).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(_i))
                _ok, _fr = cap.read()
                if not _ok:
                    continue
                _K = _fov.get_cam_intrinsics(cv2.cvtColor(_fr, cv2.COLOR_BGR2RGB))
                if hasattr(_K, "detach"):          # tensor torch
                    _K = _K.detach().float().cpu().numpy()
                _K = np.asarray(_K, dtype=np.float64).reshape(3, 3)
                _fx_s.append(_K[0, 0])
                _fy_s.append(_K[1, 1])
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # la boucle repart de zéro
            if _fx_s:
                _fx_m, _fy_m = float(np.median(_fx_s)), float(np.median(_fy_s))
                cam_int = torch.tensor(
                    [[_fx_m, 0.0, width / 2.0], [0.0, _fy_m, height / 2.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32).unsqueeze(0).cuda()
                _sp = float(np.std(_fx_s)) if len(_fx_s) > 1 else 0.0
                print(f"  [fov_once] fx={_fx_m:.1f} fy={_fy_m:.1f} "
                      f"(médiane de {len(_fx_s)}/{_n} frames, écart-type fx={_sp:.1f}, "
                      f"{time.time() - _t_fov:.2f}s) → MoGe ne tournera plus par frame")
            else:
                print("  [fov_once] aucune frame exploitable, repli sur l'estimation par frame")

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    prefix = f"markers_{video_name}"

    # Output paths — mirror SAM3D-OpenSim naming convention
    # vid_path garde le nom historique (_skeleton.mp4) → récupération AWS/S3
    # inchangée, mais son CONTENU est maintenant le nouvel overlay Synkro. La
    # boucle remplit d'abord une vidéo brute temporaire (_raw.mp4) qui sert de
    # fond au rendu de l'overlay (post-pass), puis est supprimée.
    vid_path       = os.path.join(args.output_dir, f"{prefix}_skeleton.mp4")
    raw_vid_path   = os.path.join(args.output_dir, f"{prefix}_raw.mp4")
    trc_path       = os.path.join(args.output_dir, f"{prefix}.trc")
    ik_mot_path    = os.path.join(args.output_dir, f"{prefix}_ik.mot")
    errors_path    = os.path.join(args.output_dir, "_ik_marker_errors.sto")
    osim_path      = os.path.join(args.output_dir, f"{prefix}_model.osim")
    mesh_glb       = os.path.join(args.output_dir, f"{prefix}_mesh.glb")
    meta_path      = os.path.join(args.output_dir, "inference_meta.json")
    outputs_path   = os.path.join(args.output_dir, "video_outputs.json")

    writer = cv2.VideoWriter(
        raw_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )

    print(f"\nVideo: {width}x{height} @ {fps:.1f}fps | {total} frames")
    print(f"Frame step: {frame_step} | Output: {args.output_dir}\n")

    t_start = time.time()

    # ── Per-frame processing ──────────────────────────────────────────────────
    frame_idx       = 0
    processed       = 0
    timestamps      = []
    all_kpts_raw    = []   # [N_frames] of [70, 3] camera-space kpts, or None
    all_kpts_2d     = []   # [N_frames] of [70, 2] image-pixel kpts, or None
                           #   (auto-validation projection → points2d.json)
    all_anat        = []   # [N_frames] of [73, 3] anatomical marker positions
                           #   (mesh vertices) camera-frame, or None → overlay
    all_focal       = []   # [N_frames] focal length (px), or None
    all_cam_t       = []   # [N_frames] of [3], or None

    # Markerset anatomique (73 marqueurs virtuels = index vertex du mesh MHR)
    # pour l'overlay interactif projeté (points2d.json, style Mesh2Sim).
    _anat_ms = _load_anatomical_markerset()
    if _anat_ms is not None:
        _anat_names, _anat_vidx, _anat_regions, _anat_bodies = _anat_ms
    else:
        _anat_names = _anat_vidx = _anat_regions = _anat_bodies = None
        print("  [points2d] correspondence_synkro.json introuvable → overlay 3D désactivé")
    all_verts       = []   # [N_frames] of [18439, 3] or None  (for mesh GLB)
    all_joint_coords = []  # [N_frames] of [127, 3] camera-space joint coords, or None
    all_raw_outputs = []   # for video_outputs.json
    # Shape-lock per-frame raw sub-params (used to regenerate meshes with a
    # locked subject morphology — see sam_3d_body/export/shape_lock.py).
    # Stored as np.ndarray or None (frame failed).
    all_shape_params     = []   # [N] of (45,) identity coeffs
    all_scale_params     = []   # [N] of (28,) PCA-encoded segment scales
    all_expr_params      = []   # [N] of (72,) face expr
    all_body_pose_params = []   # [N] of (133,) body pose (positions 124..129 = shape modes)
    all_global_rot       = []   # [N] of (3,)
    all_hand_pose_params = []   # [N] of (108,) or None
    inference_times = []
    # Multi-person track storage — keyed by track ID
    tracks = {}  # {track_id: {'kpts': [...], 'cam_t': [...], 'jcoords': [...]}}

    # Enable BoT-SORT tracking when multi_person is requested
    use_botsort = (
        getattr(args, 'multi_person', False)
        and args.tracker != "none"
        and getattr(estimator, 'detector', None) is not None
    )
    if use_botsort:
        # Use our robust config (higher track_buffer, stricter new_track_thresh)
        # to survive black/blank frames without creating spurious tracks.
        tracker_cfg_path = os.path.join(parent_dir, "tools", f"{args.tracker}_robust.yaml")
        if os.path.isfile(tracker_cfg_path):
            tracker_cfg = tracker_cfg_path
        else:
            tracker_cfg = f"{args.tracker}.yaml"
        estimator.detector.enable_tracking(tracker=tracker_cfg)
    black_frame_count = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        # Skip black/saturated frames — sending them to the tracker would
        # cause it to lose all tracks and create spurious new IDs.
        mean_brightness = np.mean(frame_bgr)
        if mean_brightness < 10:
            black_frame_count += 1
            print(f"  [{processed+1}] frame {frame_idx:5d} | SKIPPED (black frame, mean={mean_brightness:.0f})")
            # Record as missing frame so arrays stay aligned
            writer.write(frame_bgr)
            timestamps.append(frame_idx / fps)
            all_kpts_raw.append(None)
            all_kpts_2d.append(None)
            all_anat.append(None)
            all_focal.append(None)
            all_cam_t.append(None)
            all_verts.append(None)
            all_joint_coords.append(None)
            all_raw_outputs.append({"frame": f"frame_{frame_idx:06d}.jpg", "outputs": []})
            all_shape_params.append(None); all_scale_params.append(None)
            all_expr_params.append(None);  all_body_pose_params.append(None)
            all_global_rot.append(None);   all_hand_pose_params.append(None)
            if getattr(args, 'multi_person', False):
                for tr in tracks.values():
                    tr['kpts'].append(None)
                    tr['cam_t'].append(None)
                    tr['jcoords'].append(None)
                    tr['verts'].append(None)
            frame_idx += 1
            processed += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_cam_int = None
        if cam_int is not None:
            frame_cam_int = cam_int.clone()
            if args.cx is None:
                frame_cam_int[0, 0, 2] = width / 2.0
            if args.cy is None:
                frame_cam_int[0, 1, 2] = height / 2.0

        t0 = time.time()

        # Detect-then-infer / chunking logic to avoid TRT engine profile limits
        outputs = None
        frame_track_ids = None   # BoT-SORT IDs for this frame (if tracking)
        use_detect_first = (
            getattr(args, 'detect_then_infer', False)
            or (getattr(args, 'inference_batch_cap', 0) and getattr(args, 'inference_batch_cap', 0) > 0)
            or getattr(args, 'force_bboxes', False)
            or use_botsort  # tracking requires detect-first to capture IDs
        )

        if use_detect_first and getattr(estimator, 'detector', None) is not None:
            det_thr_use = getattr(args, 'bbox_thr', None) or 0.5
            det_nms_use = getattr(args, 'nms_thr', None) or 0.3
            try:
                det_res = estimator.detector.run_human_detection(
                    frame_bgr,
                    det_cat_id=0,
                    bbox_thr=det_thr_use,
                    nms_thr=det_nms_use,
                    default_to_full_image=False,
                )
            except Exception as _det_e:
                det_res = None

            boxes = None
            if det_res is not None:
                if isinstance(det_res, dict):
                    boxes = det_res.get('boxes', None)
                    frame_track_ids = det_res.get('track_ids', None)
                else:
                    boxes = det_res

            if boxes is not None and len(boxes) > 0:
                boxes = np.asarray(boxes)
                # Sort by area (largest first) so we keep main subjects when limiting
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                order = np.argsort(-areas)
                boxes = boxes[order]
                if frame_track_ids is not None:
                    frame_track_ids = np.asarray(frame_track_ids)[order]

                max_p = getattr(args, 'max_persons', None)
                if max_p is not None and max_p > 0:
                    boxes = boxes[:max_p]
                    if frame_track_ids is not None:
                        frame_track_ids = frame_track_ids[:max_p]

                batch_cap = getattr(args, 'inference_batch_cap', 0) or 0
                # If force_bboxes and batch_cap==0, treat as single large batch
                if getattr(args, 'force_bboxes', False) and batch_cap == 0:
                    batch_cap = boxes.shape[0]

                # Single-call if no chunking needed
                if batch_cap <= 0 or batch_cap >= boxes.shape[0]:
                    try:
                        outputs = estimator.process_one_image(
                            frame_rgb,
                            bboxes=boxes,
                            hand_box_source=args.hand_box_source,
                            inference_type=args.inference_type,
                            cam_int=frame_cam_int,
                        )
                        print(f"  [detect_then_infer] processed {len(boxes)} boxes in one batch")
                    except Exception as e:
                        print(f"  [detect_then_infer] inference failed: {e}")
                        outputs = None
                else:
                    # Chunked inference to avoid exceeding TRT batch profiles
                    outputs = []
                    for i in range(0, len(boxes), batch_cap):
                        chunk = boxes[i : i + batch_cap]
                        try:
                            outs_chunk = estimator.process_one_image(
                                frame_rgb,
                                bboxes=np.asarray(chunk),
                                hand_box_source=args.hand_box_source,
                                inference_type=args.inference_type,
                                cam_int=frame_cam_int,
                            )
                            if outs_chunk:
                                outputs.extend(outs_chunk)
                            print(f"  [detect_then_infer] chunk {i//batch_cap+1} processed {len(chunk)} boxes")
                        except Exception as e_chunk:
                            print(f"  [detect_then_infer] chunk inference failed (size {len(chunk)}): {e_chunk}")
                    if len(outputs) == 0:
                        outputs = None

        # If we still have no outputs, fall back to the original top-down call.
        # Skip fallback when BoT-SORT tracking is active — calling the detector
        # again would corrupt the tracker's internal state.
        if outputs is None and not use_botsort:
            try:
                outputs = estimator.process_one_image(
                    frame_rgb,
                    hand_box_source=args.hand_box_source,
                    inference_type=args.inference_type,
                    cam_int=frame_cam_int,
                    bbox_thr=getattr(args, 'bbox_thr', None) or 0.5,
                    nms_thr=getattr(args, 'nms_thr', None) or 0.3,
                )
            except Exception as e:
                print(f"  Frame {frame_idx}: inference error — {e}")
                writer.write(frame_bgr)
                timestamps.append(frame_idx / fps)
                all_kpts_raw.append(None)
                all_kpts_2d.append(None)
                all_anat.append(None)
                all_focal.append(None)
                all_cam_t.append(None)
                all_verts.append(None)
                all_joint_coords.append(None)
                all_raw_outputs.append({"frame": f"frame_{frame_idx:06d}.jpg", "outputs": []})
                all_shape_params.append(None); all_scale_params.append(None)
                all_expr_params.append(None);  all_body_pose_params.append(None)
                all_global_rot.append(None);   all_hand_pose_params.append(None)
                frame_idx += 1
                processed += 1
                continue

        # If detector found bboxes but some outputs lack pose estimates, try a forced per-bbox pass.
        # Skip when tracking is active — re-calling the detector would corrupt tracker state.
        if not use_botsort:
            try:
                need_forced = False
                if outputs:
                    for p in outputs:
                        if 'pred_keypoints_3d' not in p or p.get('pred_keypoints_3d') is None:
                            need_forced = True
                            break
                if need_forced and getattr(estimator, 'detector', None) is not None:
                    print("  [fallback] Some detections missing pose -> running detector + forced per-bbox inference...")
                    for p in outputs:
                        if 'bbox' in p and ('pred_keypoints_3d' not in p or p.get('pred_keypoints_3d') is None):
                            b = p['bbox']
                            w = max(0.0, b[2] - b[0])
                            h = max(0.0, b[3] - b[1])
                            area = w * h
                            cx, cy = _centroid_from_bbox(b)
                            print(f"    [missing] bbox={b} area={area:.1f} centroid=({cx:.1f},{cy:.1f})")

                    det_thr = getattr(args, 'bbox_thr_force', 0.2)
                    det_nms = getattr(args, 'nms_thr', 0.3)
                    try:
                        det_res = estimator.detector.run_human_detection(
                            frame_bgr,
                            det_cat_id=0,
                            bbox_thr=det_thr,
                            nms_thr=det_nms,
                            default_to_full_image=False,
                        )
                    except Exception as _det_e:
                        det_res = None
                    if det_res is not None:
                        if isinstance(det_res, dict):
                            boxes = det_res.get('boxes', None)
                        else:
                            boxes = det_res
                        if boxes is not None and len(boxes) > 0:
                            try:
                                forced_outs = estimator.process_one_image(
                                    frame_rgb,
                                    bboxes=np.asarray(boxes),
                                    hand_box_source=args.hand_box_source,
                                    inference_type=args.inference_type,
                                    cam_int=frame_cam_int,
                                )
                                if forced_outs:
                                    outputs = forced_outs
                                    frame_out = {"frame": f"frame_{frame_idx:06d}.jpg", "outputs": []}
                                    for p in outputs:
                                        entry: dict = {}
                                        if "bbox" in p:
                                            entry["bbox"] = [float(x) for x in p["bbox"]]
                                        if "pred_cam_t" in p and p["pred_cam_t"] is not None:
                                            entry["focal_length"] = float(p.get("focal_length", 0.0))
                                        if "pred_keypoints_3d" in p and p["pred_keypoints_3d"] is not None:
                                            entry["pred_keypoints_3d"] = p["pred_keypoints_3d"].tolist()
                                        frame_out["outputs"].append(entry)
                                    all_raw_outputs[-1] = frame_out
                                    print(f"  [fallback] forced per-bbox inference produced {len(outputs)} outputs")
                            except Exception as e_forced:
                                print(f"  [fallback] forced per-bbox inference failed: {e_forced}")
            except Exception:
                pass

        # Handle no outputs (e.g. no person detected this frame)
        if outputs is None:
            outputs = []

        # Attach BoT-SORT track IDs to each output person dict
        if frame_track_ids is not None and len(outputs) > 0:
            for pi, p in enumerate(outputs):
                if pi < len(frame_track_ids):
                    p['track_id'] = int(frame_track_ids[pi])

        inf_t = time.time() - t0
        inference_times.append(inf_t)

        # Pick the largest person (most confident detection)
        person = outputs[0] if outputs else None

        # Collect raw keypoints, camera translation, and joint coords for this frame
        if person is not None:
            kpts  = person.get("pred_keypoints_3d")   # [70, 3] camera space
            cam_t = person.get("pred_cam_t")           # [3]
            if kpts is not None and cam_t is not None and not np.any(np.isnan(kpts)):
                all_kpts_raw.append(kpts.copy())
                all_cam_t.append(cam_t.copy())
            else:
                all_kpts_raw.append(None)
                all_cam_t.append(None)
            # Keypoints 2D projetés (pixels image) → auto-validation projection.
            k2 = person.get("pred_keypoints_2d")       # [70, 2] pixels, ou None
            if k2 is not None and not np.any(np.isnan(k2)):
                all_kpts_2d.append(np.asarray(k2)[:, :2].copy())
            else:
                all_kpts_2d.append(None)
            # Marqueurs anatomiques (73 index vertex du mesh) → overlay projeté.
            all_focal.append(person.get("focal_length"))
            _vrt = person.get("pred_vertices")          # [18439, 3] repère réseau
            if (_anat_vidx is not None and _vrt is not None
                    and not np.any(np.isnan(_vrt))):
                all_anat.append(np.asarray(_vrt)[_anat_vidx].astype(np.float32).copy())
            else:
                all_anat.append(None)
            jc = person.get("pred_joint_coords")       # [127, 3] camera space
            if jc is not None and not np.any(np.isnan(jc)):
                all_joint_coords.append(jc.copy())
            else:
                all_joint_coords.append(None)
            # Shape-lock raw sub-params (always copied — None only if absent).
            sp = person.get("shape_params");      all_shape_params.append(np.asarray(sp).copy() if sp is not None else None)
            sc = person.get("scale_params");      all_scale_params.append(np.asarray(sc).copy() if sc is not None else None)
            ep = person.get("expr_params");       all_expr_params.append(np.asarray(ep).copy() if ep is not None else None)
            bp = person.get("body_pose_params");  all_body_pose_params.append(np.asarray(bp).copy() if bp is not None else None)
            gr = person.get("global_rot");        all_global_rot.append(np.asarray(gr).copy() if gr is not None else None)
            hp = person.get("hand_pose_params");  all_hand_pose_params.append(np.asarray(hp).copy() if hp is not None else None)
        else:
            all_kpts_raw.append(None)
            all_kpts_2d.append(None)
            all_anat.append(None)
            all_focal.append(None)
            all_cam_t.append(None)
            all_joint_coords.append(None)
            all_shape_params.append(None); all_scale_params.append(None)
            all_expr_params.append(None);  all_body_pose_params.append(None)
            all_global_rot.append(None);   all_hand_pose_params.append(None)

        timestamps.append(frame_idx / fps)

        # Collect raw outputs for video_outputs.json
        frame_out = {"frame": f"frame_{frame_idx:06d}.jpg", "outputs": []}
        for p in outputs:
            entry: dict = {}
            if "bbox" in p:
                entry["bbox"] = [float(x) for x in p["bbox"]]
            if "pred_cam_t" in p and p["pred_cam_t"] is not None:
                entry["focal_length"] = float(p.get("focal_length", 0.0))
            if "pred_keypoints_3d" in p and p["pred_keypoints_3d"] is not None:
                entry["pred_keypoints_3d"] = p["pred_keypoints_3d"].tolist()
            frame_out["outputs"].append(entry)
        all_raw_outputs.append(frame_out)

        # --- Multi-person tracking ---
        if getattr(args, 'multi_person', False):
            # Advance all existing tracks with a placeholder for this frame
            for tr in tracks.values():
                tr['kpts'].append(None)
                tr['cam_t'].append(None)
                tr['jcoords'].append(None)
                tr['verts'].append(None)

            for p in outputs:
                # Get track ID: from BoT-SORT when available, otherwise
                # fall back to centroid-based matching (--tracker none).
                tid = p.get('track_id', -1)
                if tid < 0 and 'bbox' in p:
                    # Centroid fallback for --tracker none
                    centroid = _centroid_from_bbox(p['bbox'])
                    best_tid, best_d = -1, 200.0
                    for existing_tid, tr in tracks.items():
                        if tr.get('_last_centroid') is None:
                            continue
                        d = _euclidean(centroid, tr['_last_centroid'])
                        if d < best_d:
                            best_d = d
                            best_tid = existing_tid
                    if best_tid >= 0:
                        tid = best_tid
                    else:
                        tid = max(tracks.keys(), default=0) + 1
                    p['track_id'] = tid

                if tid < 0:
                    continue
                if tid not in tracks:
                    tracks[tid] = {
                        'id': tid,
                        'kpts': [None] * processed + [None],
                        'cam_t': [None] * processed + [None],
                        'jcoords': [None] * processed + [None],
                        'verts': [None] * processed + [None],
                        'bbox_cx': [],  # image-space center X for left→right sorting
                    }
                kpts_p = p.get('pred_keypoints_3d')
                cam_t_p = p.get('pred_cam_t')
                jc_p = p.get('pred_joint_coords')
                tracks[tid]['kpts'][-1] = kpts_p.copy() if kpts_p is not None else None
                tracks[tid]['cam_t'][-1] = cam_t_p.copy() if cam_t_p is not None else None
                tracks[tid]['jcoords'][-1] = jc_p.copy() if jc_p is not None else None
                # Collect mesh vertices for per-person GLB export
                verts_p = p.get('pred_vertices')
                if verts_p is not None and cam_t_p is not None and not np.any(np.isnan(verts_p)):
                    tracks[tid]['verts'][-1] = (verts_p + cam_t_p[None, :]).astype(np.float32)
                if 'bbox' in p:
                    cx, _ = _centroid_from_bbox(p['bbox'])
                    tracks[tid]['_last_centroid'] = (cx, _centroid_from_bbox(p['bbox'])[1])
                    tracks[tid]['bbox_cx'].append(cx)

        # Collect mesh vertices for mesh GLB and/or --markerset flodelaplace
        # (the latter needs the 21 anatomical vertex positions per frame).
        need_verts = (not args.no_mesh_glb) or force_collect_verts or args.export_mesh_npz
        if need_verts and person is not None:
            verts = person.get("pred_vertices")
            cam_t = person.get("pred_cam_t")
            if verts is not None and cam_t is not None and not np.any(np.isnan(verts)):
                # Store RAW verts (no cam_t) — apply_pipeline_to_verts will add
                # the XZ delta of cam_t internally via _last_xz_deltas_m, exactly
                # the way transform() does it for kpts. Adding cam_t here baked
                # a residual cam_t.Y into the mesh frame that did not exist in
                # the kpts/anatomical chain, putting the mesh visibly below
                # ground for shots where the camera sits well above the subject
                # (e.g. rower seated on machine).
                all_verts.append(verts.astype(np.float32))
            else:
                all_verts.append(None)
        else:
            all_verts.append(None)

        # On écrit la frame BRUTE (l'overlay Synkro est rendu en post-pass depuis
        # points2d.json → markers_*_overlay.mp4). Fini l'ancien squelette cuit.
        writer.write(frame_bgr)

        processed += 1
        avg_fps = 1.0 / (sum(inference_times) / len(inference_times))
        detected = len(outputs)
        print(
            f"  [{processed}/{total // frame_step}] frame {frame_idx:5d} | "
            f"{inf_t:.2f}s | {detected} person(s) | avg {avg_fps:.2f} fps",
            flush=True,
        )

        frame_idx += 1

    cap.release()
    writer.release()

    # points2d.json (overlay interactif front-end) — MARQUEURS ANATOMIQUES
    # (73 marqueurs virtuels) projetés à l'écran, style Mesh2Sim, colorés par
    # région. Alignés frame-pour-frame avec markers_*_skeleton.mp4. Écrit dans
    # analytics/ (à côté de metrics.json), indépendant de --module.
    if _anat_names is None:
        # Pas de markerset → pas d'overlay : garde la vidéo brute sous le nom
        # historique pour ne rien casser en aval.
        try: os.replace(raw_vid_path, vid_path)
        except OSError: pass
    if _anat_names is not None:
        try:
            _label_set = {"HTOP", "C7", "RACR", "LACR", "XIPH", "RGTR", "LGTR",
                          "RASI", "LASI", "RPSI", "LPSI", "RLEL", "RMEL", "LLEL",
                          "LMEL", "RLFC", "RMFC", "LLFC", "LMFC", "RLMAL", "RMMAL",
                          "LLMAL", "LMMAL", "RTOE", "LTOE", "RCAL", "LCAL"}
            # Map nom keypoint mhr70 → id (pour projeter les centres articulaires).
            _kpt_name_to_id = {v["name"]: v["id"]
                               for v in mhr70_pose_info.get("keypoint_info", {}).values()}
            pts2d_path = os.path.join(args.output_dir, "analytics", "points2d.json")
            _nf, _nk, _ne, _err = write_points2d_json_3d(
                all_anat, all_cam_t, all_focal, _anat_names, _anat_regions,
                _label_set, width, height, out_fps, pts2d_path,
                all_kpts_raw=all_kpts_raw, all_kpts_2d=all_kpts_2d,
                kpt_name_to_id=_kpt_name_to_id)
            _errmsg = (f"reproj médiane={_err:.2f}px" if _err is not None
                       else "reproj n/a")
            print(f"points2d.json → {_nf} frames × {_nk} points (marqueurs + JC) "
                  f"+ {_ne} segments squelette ({width}x{height} @ {out_fps:.1f}fps, "
                  f"offset_s=0, {_errmsg}) → {pts2d_path}")
            # Overlay Synkro baké (nouveau) → markers_*_skeleton.mp4 (même nom
            # qu'avant pour l'AWS), rendu depuis points2d.json sur la vidéo brute.
            # La brute temporaire est supprimée si le rendu réussit.
            try:
                _no = render_overlay_video(pts2d_path, raw_vid_path, vid_path)
                print(f"overlay Synkro → {vid_path} ({_no} frames)")
                if os.path.isfile(vid_path) and os.path.getsize(vid_path) > 0:
                    try: os.remove(raw_vid_path)
                    except OSError: pass
            except Exception as _eo:
                print(f"WARN: overlay baké échec ({_eo}) — fallback vidéo brute")
                try: os.replace(raw_vid_path, vid_path)  # au moins la brute sous le bon nom
                except OSError: pass
        except Exception as _e:
            print(f"WARN: points2d.json (3D) échec — {_e}")
            try: os.replace(raw_vid_path, vid_path)  # pas de points2d → garde la brute
            except OSError: pass

    print(f"\nDone! {processed} frames processed.")
    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print(f"Avg inference: {avg:.2f}s/frame ({1/avg:.2f} fps)")

    # ── Shape-lock : médiane des paramètres morpho sur tout l'essai, puis ─────
    # régénération des vertices/joint_coords/keypoints avec ce shape locké +
    # la pose conservée per-frame. Désactivable via --no_shape_lock.
    # En multi-person on skip pour l'instant (un shape par track serait
    # nécessaire ; pas implémenté).
    _do_shape_lock = (
        (not args.no_shape_lock)
        and (not getattr(args, 'multi_person', False))
    )
    if _do_shape_lock:
        try:
            from sam_3d_body.export.shape_lock import (
                aggregate_shape, regenerate_with_locked_shape,
            )
            n_valid = sum(1 for x in all_body_pose_params if x is not None)
            if n_valid == 0:
                print("[shape_lock] No valid frame to aggregate — skipping.")
            else:
                print(f"[shape_lock] Aggregating subject shape across {n_valid} frames (median)...")
                locked = aggregate_shape(
                    all_shape_params, all_scale_params,
                    all_expr_params, all_body_pose_params,
                )
                # Sauvegarde des betas morpho verrouillés (ajout de sortie pur).
                # Vecteur 73 = [identité(45), scale(28)] — c'est exactement ce
                # qu'attend Mesh2Marker pour générer un .osim à la morphologie
                # du sujet (marqueurs posés sur SA peau).
                try:
                    _bs = locked.get("shape_params"); _bsc = locked.get("scale_params")
                    if _bs is not None and _bsc is not None:
                        _betas_path = os.path.join(args.output_dir, "_shape_betas.npz")
                        np.savez(_betas_path,
                                 betas73=np.concatenate([np.asarray(_bs).ravel(),
                                                         np.asarray(_bsc).ravel()]),
                                 shape_params=np.asarray(_bs).ravel(),
                                 scale_params=np.asarray(_bsc).ravel())
                        print(f"[shape_lock] betas → {_betas_path}")
                except Exception as _e:
                    print(f"[shape_lock] sauvegarde betas échouée: {_e}")
                per_frame_pose = []
                for i in range(len(all_body_pose_params)):
                    if (all_body_pose_params[i] is None
                        or all_cam_t[i] is None
                        or all_global_rot[i] is None):
                        per_frame_pose.append(None)
                    else:
                        per_frame_pose.append({
                            "global_trans":      all_cam_t[i],
                            "global_rot":        all_global_rot[i],
                            "body_pose_params":  all_body_pose_params[i],
                            "hand_pose_params":  all_hand_pose_params[i],
                        })
                n_regen = sum(1 for x in per_frame_pose if x is not None)
                print(f"[shape_lock] Regenerating {n_regen} frames via mhr_head._mhr_forward_core...")
                regen = regenerate_with_locked_shape(
                    estimator.model.head_pose, locked, per_frame_pose,
                    device="cuda",
                )
                n_replaced = 0
                for i, r in enumerate(regen):
                    if r is None:
                        continue
                    if i < len(all_verts) and all_verts[i] is not None:
                        all_verts[i] = r["pred_vertices"].astype(np.float32)
                    if i < len(all_joint_coords) and all_joint_coords[i] is not None:
                        all_joint_coords[i] = r["pred_joint_coords"].astype(np.float32)
                    if (i < len(all_kpts_raw) and all_kpts_raw[i] is not None
                        and r["pred_keypoints_3d"] is not None):
                        new_k = r["pred_keypoints_3d"].astype(np.float32)
                        if new_k.shape == all_kpts_raw[i].shape:
                            all_kpts_raw[i] = new_k
                    n_replaced += 1
                print(f"[shape_lock] Replaced verts/jcoords/kpts on {n_replaced} frames.")
        except Exception as e:
            print(f"[shape_lock] FAILED: {e}. Falling back to raw per-frame shape.")
            import traceback
            traceback.print_exc()

    # ── Build raw keypoint and jcoords arrays (NaN for missing frames) ─────────
    N = len(timestamps)
    kpts_stack    = np.full((N, 70,  3), np.nan, dtype=np.float64)
    cam_t_stack   = np.full((N, 3),      np.nan, dtype=np.float64)
    jcoords_stack = np.full((N, 127, 3), np.nan, dtype=np.float64)
    for i, (k, t_) in enumerate(zip(all_kpts_raw, all_cam_t)):
        if k is not None and t_ is not None:
            kpts_stack[i]  = k
            cam_t_stack[i] = t_
    for i, jc in enumerate(all_joint_coords):
        if jc is not None:
            jcoords_stack[i] = jc

    good = int(np.sum(~np.any(np.isnan(kpts_stack), axis=(1, 2))))
    print(f"\n  Frames with detected person: {good}/{processed}")

    if good == 0:
        print("  No valid frames — skipping OpenSim export.")
        return

    # ── Post-processing pipeline ───────────────────────────────────────────────
    subject_height = args.person_height if args.person_height is not None else 1.75

    print("\nPost-processing keypoints...")

    # 1. Interpolate missing frames and Butterworth-filter (no manual bone scaling —
    #    the estimator's proportions are used directly; height is set from user input)
    post_proc = PostProcessor()
    kpts_processed    = post_proc.process(kpts_stack, fps=out_fps)
    jcoords_processed = post_proc.process_jcoords(jcoords_stack, fps=out_fps)

    # 1b. --markerset flodelaplace: pull the 21 anatomical mesh vertex positions
    # per frame and append them to the jcoords array so they ride the same
    # post-proc / transform / lean-correction pipeline. Split back out before
    # calling the Florian converter.
    #
    # Frame alignment: since the cam_t fix below, `all_verts[i]` is stored
    # RAW (no cam_t), in the same camera-local frame as jcoords_stack. So we
    # can extract anatomical verts directly without subtracting cam_t.
    N_anat = 0
    if markerset == "flodelaplace":
        N_anat = len(florian_converter.vertex_indices)
        anat_stack = np.full((N, N_anat, 3), np.nan, dtype=np.float64)
        for i, v in enumerate(all_verts):
            if v is None:
                continue
            anat_stack[i] = florian_converter.extract_anatomical(v)
        anat_processed = post_proc.process_jcoords(anat_stack, fps=out_fps)
        jcoords_processed = np.concatenate([jcoords_processed, anat_processed], axis=1)
    # Interpolate + smooth cam_t for global walking trajectory in TRC.
    # Reshape to (N,1,3) so PostProcessor's per-keypoint logic handles it,
    # then squeeze back to (N,3).
    cam_t_processed = post_proc.process_jcoords(
        cam_t_stack[:, np.newaxis, :], fps=out_fps
    )[:, 0, :]

    # 2. Rotate axes (camera → OpenSim Y-up), scale to subject_height using
    #    c_head (jcoords[113]) as the exact top reference — no magic constants.
    #    Global XZ trajectory comes from cam_t (only XZ is applied, not Y).
    transformer = CoordinateTransformer(subject_height=subject_height)
    # `--floor` flag : si présent, active la mise au sol (per-frame) et le
    # redressement (one-shot floor lean correction). Si absent, le sujet reste
    # dans sa position 3D réelle (utile pour rameur, couché, suspension, etc.).
    _apply_floor = args.floor or args.floor_seated
    # --floor_seated désactive uniquement le body-vertical correction (le
    # sujet assis n'a pas un axe midfoot→neck vertical à imposer).
    _apply_body_vertical = (not args.floor_seated) if _apply_floor else None
    # correct_floor_lean est DÉCOUPLÉ de align_to_ground :
    # - Activé si MoGe a calculé un angle (= --floor_moge) ou si --floor
    # - Permet d'avoir le redressement caméra (pitch/roll/body-vertical) en
    #   mode défaut (--floor_moge sans --floor), sans forcer la mise au sol
    #   per_frame qui est l'objet propre de --floor.
    _correct_lean = (moge_floor_angle is not None or _apply_floor) and not args.no_lean_fix
    # Auto-détection selon --module :
    # - running/gait/sprint : subject avance en ligne droite → --lock_lateral
    # - squat/STS : feet stables → --feet_anchor + clamp (feet à Y≥0 auto)
    # - CMJ/jump : feet peuvent lever pendant vol → --feet_anchor SANS clamp
    #   (le clamp shift everybody up quand MHR bugge le crouch → inverse la
    #    biomeca du saut). NO_FLOOR_CLAMP=1 propagé.
    # - cycling : assis pédale → --stationary + --lock_vertical
    _auto_lock_lateral = args.module in ("d3.running", "d3.gait", "d3.sprint_start")
    # Activités "en place" (subject bouge peu en XZ globalement) :
    # combo --stationary + --feet_anchor + clamp Y≥0.
    #   • --stationary → active injection cam_t.Y (nécessaire pour capter la
    #     motion verticale que MHR sous-reconstruit sur CMJ/squat/STS).
    #   • --feet_anchor → shift XZ post pour que le midpoint pieds reste au
    #     médian → effectivement les PIEDS sont locked (pas le pelvis).
    #   • clamp Y≥0 (défaut floor_moge) → empêche traversée sol.
    _auto_feet_anchor = args.module in ("d3.squat", "d3.sit_to_stand",
                                          "d3.jump")
    _auto_stationary = args.module in ("d3.squat", "d3.sit_to_stand",
                                          "d3.jump", "d3.cycling")
    # MODE TAPIS AUTO : dès qu'une vitesse tapis est fournie (token tm<kmh>) sur
    # course/marche, le sujet est EN PLACE → --stationary récupère l'oscillation
    # verticale (rebond) via l'injection cam_t.Y, sinon le bassin reste plat et
    # ça "glisse" au lieu de courir. L'anti-skate est désactivé en aval (le pied
    # recule avec la bande, il DOIT glisser). Le stride est déjà calculé via
    # vitesse × temps d'appui côté module.
    _is_treadmill = (getattr(args, "treadmill_speed", None) is not None
                     and args.module in ("d3.running", "d3.gait"))
    if _is_treadmill:
        _auto_stationary = True
        print(f"  [mode tapis] vitesse {args.treadmill_speed} m/s → --stationary "
              f"auto (rebond vertical) + anti-skate OFF (pied glisse avec la bande)")
    _auto_lock_vertical = args.module == "d3.cycling"
    # Tests RTS unipodaux : single_leg_squat (statique pied au sol) et
    # single_leg_hop (sauts avec vol). L'ancrage conscient du contact gère les
    # deux — bassin qui descend en SLS, vol préservé en hop — sans le pré-réglage
    # feet_anchor/stationary qui casserait la phase de vol.
    _auto_contact_anchor = args.module in ("d3.single_leg_squat",
                                            "d3.single_leg_hop")
    _stationary_effective = args.stationary or _auto_stationary
    # Running/gait/sprint : feet peuvent lever naturellement → clamp désactivé.
    # Écrit à CHAQUE vidéo, y compris "0" : sinon la valeur posée par une vidéo
    # de course resterait active pour les suivantes (squat, CMJ, STS, cycling)
    # dans un process qui en traite plusieurs → marqueurs pieds décalés de
    # plusieurs cm, TRC et IK faux, sans le moindre message d'erreur.
    os.environ["NO_FLOOR_CLAMP"] = (
        "1" if args.module in ("d3.running", "d3.gait", "d3.sprint_start") else "0"
    )

    kpts_opensim, jcoords_opensim = transformer.transform(
        kpts_processed,
        jcoords_3d=jcoords_processed,
        camera_translation=cam_t_processed,
        center_pelvis=True,
        align_to_ground=_apply_floor,
        apply_global_translation=not _stationary_effective,
        correct_floor_lean=_correct_lean,
        floor_angle=moge_floor_angle,
        apply_body_vertical=_apply_body_vertical,
        lock_vertical=args.lock_vertical,
        lock_lateral=args.lock_lateral or _auto_lock_lateral,
        contact_anchor=args.contact_anchor or _auto_contact_anchor,
        fps=out_fps,
    )

    # 2b. Spine-based forward-lean correction (runs after floor-plane rotation above).
    if args.lean_angle is not None:
        # Manual override — use exactly this angle
        print(f"  [lean] manual correction {args.lean_angle:+.2f}°")
        kpts_opensim, jcoords_opensim = transformer.correct_forward_lean(
            kpts_opensim, jcoords=jcoords_opensim, angle=args.lean_angle
        )
    elif args.lean_ref_frame is not None and not args.no_lean_fix:
        # Reference frame: measure spine lean over a ±5 frame window around
        # the frame where the person is known to be standing upright.
        # Averaging makes it robust to single-frame keypoint noise.
        ref = min(args.lean_ref_frame, kpts_opensim.shape[0] - 1)
        lean_angle = _lean_angle_over_range(kpts_opensim, ref)
        print(f"  [spine lean] ref frame {ref} (±5 avg): measured {lean_angle:+.2f}° → correcting")
        kpts_opensim, jcoords_opensim = transformer.correct_forward_lean(
            kpts_opensim, jcoords=jcoords_opensim, angle=lean_angle
        )
    elif not args.no_lean_fix and getattr(args, 'enable_auto_lean_fix', False):
        # Mode auto : DÉSACTIVÉ PAR DÉFAUT (cf. --enable_auto_lean_fix).
        # La mesure spine sur kpts acromions (67, 68) est biaisée par MHR
        # — anatomical termine plus penché que mesh. À activer seulement
        # sur les vidéos où c'est calibré et vérifié visuellement.
        src = " (after MoGe)" if moge_floor_angle is not None else ""
        used_static_ref = False
        if getattr(args, "auto_static_calib", True):
            try:
                _ts, _te, f_s, f_e = _detect_static_window(kpts_opensim, out_fps)
                ref = (f_s + f_e) // 2
                lean_angle = _lean_angle_over_range(kpts_opensim, ref)
                print(f"  [spine lean] auto-ref frame {ref} "
                      f"(milieu de la quietest window {f_s}-{f_e}): "
                      f"measured {lean_angle:+.2f}°{src} → correcting")
                used_static_ref = True
            except Exception as err:
                print(f"  [spine lean] _detect_static_window failed ({err}), "
                      f"falling back to median estimator")
        if not used_static_ref:
            lean_angle = transformer._estimate_lean_angle(kpts_opensim)
            print(f"  [spine lean] estimated {lean_angle:+.2f}°{src} "
                  f"(median all frames) → correcting")
        kpts_opensim, jcoords_opensim = transformer.correct_forward_lean(
            kpts_opensim, jcoords=jcoords_opensim, angle=lean_angle
        )

    # 2c. Camera-pitch-based lean correction (experimental, opt-in)
    if getattr(args, 'lean_cam_pitch_fix', False):
        pitch_angle = transformer._estimate_pitch_angle(cam_t_processed)
        print(f"  Camera pitch correction: {pitch_angle:.2f}°")
        kpts_opensim, jcoords_opensim = transformer.correct_lean_cam_pitch(
            kpts_opensim, jcoords=jcoords_opensim, cam_t=cam_t_processed
        )

    # 3. Map MHR70 → OpenSim marker names.
    body_only = (args.inference_type == "body")
    if markerset == "flodelaplace":
        # Split anatomical verts back out of the extended jcoords array.
        anat_verts_opensim = jcoords_opensim[:, 127:]
        jcoords_opensim    = jcoords_opensim[:, :127]
        markers_array, marker_names = florian_converter.convert(
            kpts_opensim, jcoords_opensim, anat_verts_opensim
        )
        # GLB skeleton overlay uses the legacy body_only layout — always build
        # it from the KeypointConverter for visual parity with the pose2sim
        # flow, even when the TRC is written with Florian names.
        markers_body, _ = KeypointConverter().convert(
            kpts_opensim, include_derived=True, body_only=True
        )
    else:
        converter = KeypointConverter()
        markers_array, marker_names = converter.convert(
            kpts_opensim, jcoords_3d=jcoords_opensim, include_derived=True, body_only=body_only
        )
        # Body-only markers for GLB skeleton visualisation (no spine appended here —
        # the GLB path builds its own spine overlay from frames_joint_coords)
        markers_body, _ = converter.convert(
            kpts_opensim, include_derived=True, body_only=True
        )

    # ── Anti-glisse pied : fige le XZ des marqueurs pieds pendant le contact,
    # sur le TRC FINAL (markerset Flodelaplace). Le TRC anti-slidé nourrit
    # ensuite IK + retargeting avatar → post-IK et avatar déjà anti-slidés (pas
    # de lavage par l'IK). ACTIVÉ PAR DÉFAUT (indépendant de --contact_anchor,
    # qui lui gère l'ancrage vertical du bassin) : dès qu'un pied touche le sol,
    # il ne doit pas déraper. Désactivable via --no_anti_foot_skate.
    _antiskate_shifts = None
    # Articulations de main : SAM3D les fournit nommees (4 par doigt), la ou le
    # markerset ne portait que des points de PEAU — et, pour le majeur et
    # l'annulaire, des positions INTERPOLEES (23,5 mm d'ecart mesure). On les
    # ajoute au tableau de marqueurs, donc au TRC.
    from sam_3d_body.export.hand_keypoints import append_hand_keypoints
    markers_array, marker_names = append_hand_keypoints(
        markers_array, marker_names, kpts_opensim)

    _anti_skate_on = (not getattr(args, "no_anti_foot_skate", False)) or args.contact_anchor
    # Mode tapis : le pied recule AVEC la bande → il DOIT glisser. On force
    # l'anti-skate OFF (sauf si --contact_anchor explicitement demandé).
    if _is_treadmill and not args.contact_anchor:
        _anti_skate_on = False
    if _anti_skate_on and marker_names is not None:
        from sam_3d_body.export.coordinate_transform import anti_foot_skate_markers
        markers_array, _antiskate_shifts = anti_foot_skate_markers(
            markers_array, marker_names, fps=out_fps)
        print("  [anti-skate] gel XZ des pieds en contact appliqué au TRC final")

    # ── --feet_anchor : shift global per-frame pour que le midpoint des
    # pieds reste à sa position médiane sur toute la vidéo. Translate tout
    # le corps (mesh + kpts + jcoords + markers) du même delta XZ par frame.
    # Effet : pieds collés au sol, le reste du corps articule autour.
    feet_anchor_shifts_xz = None  # (N, 2) array, [dx, dz] per frame, or None
    if args.feet_anchor or _auto_feet_anchor:
        # Référence : midpoint LCAL/RCAL (talons) si dispo, sinon LAJC/RAJC
        name_to_idx = {n: i for i, n in enumerate(marker_names)}
        ref_pair = None
        for cand in [("LCAL", "RCAL"), ("LAJC", "RAJC")]:
            if cand[0] in name_to_idx and cand[1] in name_to_idx:
                ref_pair = cand
                break
        if ref_pair is None:
            print("  [feet_anchor] WARNING: no LCAL/RCAL or LAJC/RAJC in markers — skipping.")
        else:
            li, ri = name_to_idx[ref_pair[0]], name_to_idx[ref_pair[1]]
            midfoot = 0.5 * (markers_array[:, li, :] + markers_array[:, ri, :])  # (N, 3)
            valid = ~(np.isnan(midfoot[:, 0]) | np.isnan(midfoot[:, 2]))
            if not valid.any():
                print("  [feet_anchor] WARNING: midfoot all NaN — skipping.")
            else:
                target_x = float(np.median(midfoot[valid, 0]))
                target_z = float(np.median(midfoot[valid, 2]))
                # markers_array / kpts_opensim / jcoords_opensim are all in
                # METRES (TRC exporter scales to mm at write time). Shifts in
                # metres directly.
                shifts = np.zeros((markers_array.shape[0], 2), dtype=np.float64)
                shifts[valid, 0] = target_x - midfoot[valid, 0]
                shifts[valid, 1] = target_z - midfoot[valid, 2]
                # Apply uniform XZ shift to all geometry (metres everywhere).
                markers_array[:, :, 0] += shifts[:, 0:1]
                markers_array[:, :, 2] += shifts[:, 1:2]
                kpts_opensim[:, :, 0] += shifts[:, 0:1]
                kpts_opensim[:, :, 2] += shifts[:, 1:2]
                if jcoords_opensim is not None:
                    jcoords_opensim[:, :, 0] += shifts[:, 0:1]
                    jcoords_opensim[:, :, 2] += shifts[:, 1:2]
                feet_anchor_shifts_xz = shifts  # in METRES, applied later to mesh
                print(f"  [feet_anchor] anchored midpoint {ref_pair[0]}/{ref_pair[1]} "
                      f"to ({target_x:+.3f}, {target_z:+.3f}) m. "
                      f"max shift = {np.max(np.abs(shifts)):.3f} m")

    # ---------------------------------------------------------------------
    # Multi-person per-track post-processing & export (if requested)
    # ---------------------------------------------------------------------
    per_person_trcs = []
    per_person_ik_results = []
    if getattr(args, 'multi_person', False) and len(tracks) > 0:
        if black_frame_count > 0:
            print(f"\n  Skipped {black_frame_count} black/saturated frame(s)")

        # Convert tracks dict to list and compute stats
        tracks_list = list(tracks.values())
        for tr in tracks_list:
            # Use bbox center X (image-space pixels) for left→right sorting.
            # cam_t[0] is unreliable because it's relative to the person crop.
            bbox_xs = tr.get('bbox_cx', [])
            tr['_median_x'] = float(np.median(bbox_xs)) if bbox_xs else 0.0
            tr['_valid_count'] = sum(1 for k in tr['kpts'] if k is not None)

        # Merge duplicate tracks: if two tracks have similar median X position
        # (same person re-tracked after a black frame), merge the shorter into
        # the longer one.  This handles the case where BoT-SORT still loses a
        # track despite the high track_buffer.
        merge_x_thresh = 80.0  # pixels — same-person lateral tolerance
        tracks_list.sort(key=lambda t: -t['_valid_count'])  # longest first
        merged_ids = set()
        for i, tr_a in enumerate(tracks_list):
            if tr_a['id'] in merged_ids:
                continue
            for j in range(i + 1, len(tracks_list)):
                tr_b = tracks_list[j]
                if tr_b['id'] in merged_ids:
                    continue
                if abs(tr_a['_median_x'] - tr_b['_median_x']) > merge_x_thresh:
                    continue
                # Check they don't overlap in time (both have valid data
                # on the same frame → different people, don't merge).
                overlap = False
                for fi in range(min(len(tr_a['kpts']), len(tr_b['kpts']))):
                    if tr_a['kpts'][fi] is not None and tr_b['kpts'][fi] is not None:
                        overlap = True
                        break
                if overlap:
                    continue
                # Merge tr_b into tr_a (fill gaps in tr_a with data from tr_b)
                for fi in range(min(len(tr_a['kpts']), len(tr_b['kpts']))):
                    if tr_a['kpts'][fi] is None and tr_b['kpts'][fi] is not None:
                        tr_a['kpts'][fi] = tr_b['kpts'][fi]
                        tr_a['cam_t'][fi] = tr_b['cam_t'][fi]
                        tr_a['jcoords'][fi] = tr_b['jcoords'][fi]
                        tr_a['verts'][fi] = tr_b['verts'][fi]
                merged_ids.add(tr_b['id'])
                tr_a['_valid_count'] = sum(1 for k in tr_a['kpts'] if k is not None)
                print(f"  [merge] track {tr_b['id']} → track {tr_a['id']} "
                      f"(median_x diff={abs(tr_a['_median_x'] - tr_b['_median_x']):.0f}px)")
        tracks_list = [tr for tr in tracks_list if tr['id'] not in merged_ids]

        # Sort by median bbox center X (left→right in the image)
        for tr in tracks_list:
            bbox_xs = tr.get('bbox_cx', [])
            tr['_median_x'] = float(np.median(bbox_xs)) if bbox_xs else 0.0
        tracks_list.sort(key=lambda t: t['_median_x'])

        # Filter out tracks with too few valid frames (noise)
        min_valid = max(5, processed // 10)  # at least 10% of frames or 5
        filtered = [tr for tr in tracks_list if tr['_valid_count'] >= min_valid]
        if len(filtered) < len(tracks_list):
            dropped = len(tracks_list) - len(filtered)
            print(f"  [filter] dropped {dropped} track(s) with <{min_valid} valid frames")
            tracks_list = filtered

        # Parse per-person heights (left-to-right order matches sorted tracks)
        per_person_height_list = None
        if args.person_heights is not None:
            per_person_height_list = [float(h.strip()) for h in args.person_heights.split(",")]

        print(f"\nMulti-person mode: exporting {len(tracks_list)} tracked person(s) (sorted left→right)")
        for ti, tr in enumerate(tracks_list):
            print(f"  P{ti+1}: track_id={tr['id']}, median_x={tr['_median_x']:.0f}px, "
                  f"valid_frames={tr['_valid_count']}/{processed}")
        exporter_person = TRCExporter(fps=out_fps, units="mm")
        per_person_ik_results = []
        per_person_markers = []   # for combined TRC
        per_person_names = []
        per_person_origins = []   # world-space origins (metres, OpenSim axes)
        per_person_verts_raw = []  # for combined mesh GLB (raw camera-world verts)
        # Process each track separately through the same pipeline
        for ti, tr in enumerate(tracks_list):
            # Build stacks for this track
            k_stack = np.full((N, 70, 3), np.nan, dtype=np.float64)
            cam_stack = np.full((N, 3), np.nan, dtype=np.float64)
            j_stack = np.full((N, 127, 3), np.nan, dtype=np.float64)
            for i in range(min(N, len(tr['kpts']))):
                k = tr['kpts'][i]
                if k is not None:
                    k_stack[i] = k
                ct = tr['cam_t'][i] if i < len(tr['cam_t']) else None
                if ct is not None:
                    cam_stack[i] = ct
                jc = tr['jcoords'][i] if i < len(tr['jcoords']) else None
                if jc is not None:
                    j_stack[i] = jc

            valid_frames = int(np.sum(~np.any(np.isnan(k_stack), axis=(1, 2))))
            if valid_frames == 0:
                print(f"  [person{ti+1:02d}] no valid frames — skipping")
                continue

            # Per-person height: use person_heights[ti] if available, else fallback
            if per_person_height_list is not None and ti < len(per_person_height_list):
                person_height_i = per_person_height_list[ti]
            else:
                person_height_i = subject_height

            print(f"  [person{ti+1:02d}] frames with detections: {valid_frames}/{N}, height={person_height_i:.2f}m")

            # Post-process per-person
            k_proc = post_proc.process(k_stack, fps=out_fps)
            j_proc = post_proc.process_jcoords(j_stack, fps=out_fps)
            cam_proc = post_proc.process_jcoords(cam_stack[:, np.newaxis, :], fps=out_fps)[:, 0, :]

            # Transform to OpenSim coords per-person
            tr_transformer = CoordinateTransformer(subject_height=person_height_i)
            k_open, j_open = tr_transformer.transform(
                k_proc,
                jcoords_3d=j_proc,
                camera_translation=cam_proc,
                center_pelvis=True,
                align_to_ground=True,
                apply_global_translation=not _stationary_effective,
                correct_floor_lean=not args.no_lean_fix,
                floor_angle=moge_floor_angle,
                apply_body_vertical=not args.floor_seated,
                lock_vertical=args.lock_vertical,
                lock_lateral=args.lock_lateral or _auto_lock_lateral,
                contact_anchor=args.contact_anchor or _auto_contact_anchor,
                fps=out_fps,
            )

            # Spine lean correction
            if args.lean_angle is not None:
                print(f"    [person{ti+1:02d} lean] manual {args.lean_angle:+.2f}°")
                k_open, j_open = tr_transformer.correct_forward_lean(
                    k_open, jcoords=j_open, angle=args.lean_angle
                )
            elif args.lean_ref_frame is not None and not args.no_lean_fix:
                ref = min(args.lean_ref_frame, k_open.shape[0] - 1)
                la = _lean_angle_over_range(k_open, ref)
                print(f"    [person{ti+1:02d} spine lean] ref frame {ref} (±5 avg): {la:+.2f}° → correcting")
                k_open, j_open = tr_transformer.correct_forward_lean(
                    k_open, jcoords=j_open, angle=la
                )
            elif not args.no_lean_fix and getattr(args, 'enable_auto_lean_fix', False):
                la = tr_transformer._estimate_lean_angle(k_open)
                src = " (after MoGe)" if moge_floor_angle is not None else ""
                print(f"    [person{ti+1:02d} spine lean] {la:+.2f}°{src} → correcting")
                k_open, j_open = tr_transformer.correct_forward_lean(k_open, jcoords=j_open, angle=la)

            if getattr(args, 'lean_cam_pitch_fix', False):
                pa = tr_transformer._estimate_pitch_angle(cam_proc)
                print(f"    [person{ti+1:02d} camera pitch] {pa:.2f}°")
                k_open, j_open = tr_transformer.correct_lean_cam_pitch(k_open, jcoords=j_open, cam_t=cam_proc)

            # Convert to markers and export per-person TRC
            markers_p, names_p = converter.convert(k_open, jcoords_3d=j_open, include_derived=True, body_only=body_only)
            trc_person_path = os.path.join(args.output_dir, f"{prefix}_person{ti+1:02d}.trc")
            exporter_person.export(markers_p, names_p, trc_person_path)
            per_person_trcs.append(trc_person_path)
            print(f"    Wrote per-person TRC → {trc_person_path}")
            # Store for combined multi-person TRC with world-space offsets
            _scale = tr_transformer._last_scale
            _cam_os = cam_proc @ CoordinateTransformer.CAMERA_TO_OPENSIM.T * _scale
            _cam_sm = tr_transformer._smooth_cam_t(_cam_os)
            per_person_markers.append(markers_p)
            per_person_names.append(names_p)
            per_person_origins.append(_cam_sm[0].copy())

            # Per-person mesh GLB export
            person_verts_list = tr['verts'][:N]
            per_person_verts_raw.append(person_verts_list)
            if not args.no_mesh_glb:
                has_verts = any(v is not None for v in person_verts_list)
                if has_verts:
                    glb_person_path = os.path.join(args.output_dir, f"{prefix}_person{ti+1:02d}_mesh.glb")
                    print(f"    Writing per-person mesh GLB → {glb_person_path}")
                    person_kpts_raw = tr['kpts'][:N]
                    person_cam_t_raw = tr['cam_t'][:N]
                    person_jcoords_raw = tr['jcoords'][:N]
                    write_mesh_glb(
                        glb_person_path, timestamps, person_verts_list,
                        estimator.faces,
                        frames_kpts=person_kpts_raw,
                        frames_cam_t=person_cam_t_raw,
                        frames_joint_coords=person_jcoords_raw,
                        body_only=body_only,
                    )

            # Optionally run OpenSim scale + IK per-person
            if getattr(args, 'run_ik_per_person', False):
                person_osim = os.path.join(args.output_dir, f"{prefix}_person{ti+1:02d}_model.osim")
                try:
                    if os.path.isfile(model_template):
                        shutil.copy(model_template, person_osim)
                        print(f"    Writing person model → {person_osim}")
                        scale_ok = run_scale_tool(
                            model_path=person_osim,
                            trc_path=trc_person_path,
                            scaled_model_path=person_osim,
                            subject_mass=args.subject_mass,
                            subject_height=person_height_i,
                            markerset=markerset,
                        )
                        if not scale_ok:
                            print(f"    WARNING: Scale Tool failed for person {ti+1} – running IK on unscaled model.")
                    else:
                        print(f"    WARNING: model template not found at {model_template} — skipping scale for person {ti+1}")

                    ik_person_mot = os.path.join(args.output_dir, f"{prefix}_person{ti+1:02d}_ik.mot")
                    ik_person_errors = os.path.join(args.output_dir, f"{prefix}_person{ti+1:02d}_ik_marker_errors.sto")
                    print(f"    Running OpenSim IK → {ik_person_mot}")
                    ik_ok = run_ik(
                        model_path=person_osim if os.path.isfile(person_osim) else person_osim,
                        trc_path=trc_person_path,
                        mot_path=ik_person_mot,
                        errors_path=ik_person_errors,
                        markerset=markerset,
                    )
                    per_person_ik_results.append({
                        'trc': trc_person_path,
                        'model': person_osim if os.path.isfile(person_osim) else None,
                        'ik_mot': ik_person_mot if ik_ok else None,
                        'ik_errors': ik_person_errors if ik_ok else None,
                        'ik_success': bool(ik_ok),
                    })
                    if not ik_ok:
                        print(f"    WARNING: OpenSim IK failed for person {ti+1}")
                except Exception as _eik:
                    print(f"    ERROR: per-person IK failed for person {ti+1}: {_eik}")

        # ── Combined multi-person TRC with world-space positions ──────────
        if len(per_person_markers) >= 2:
            ref_origin = per_person_origins[0]
            combined_names = []
            combined_list = []
            for pi, (mk, nm, org) in enumerate(
                zip(per_person_markers, per_person_names, per_person_origins)
            ):
                offset = org - ref_origin          # (3,) in metres
                shifted = mk.copy()
                shifted[:, :, 0] += offset[0]      # X (forward / anterior)
                shifted[:, :, 2] += offset[2]      # Z (lateral)
                combined_list.append(shifted)
                combined_names.extend([f"P{pi+1}_{n}" for n in nm])
            combined_markers_all = np.concatenate(combined_list, axis=1)
            combined_trc_path = os.path.join(
                args.output_dir, f"{prefix}_combined.trc"
            )
            TRCExporter(fps=out_fps, units="mm").export(
                combined_markers_all, combined_names, combined_trc_path
            )
            per_person_trcs.append(combined_trc_path)
            print(
                f"\n    Combined multi-person TRC ({len(per_person_markers)} persons)"
                f" → {combined_trc_path}"
            )

        # ── Combined multi-person mesh GLB (all people in one scene) ─────
        if not args.no_mesh_glb and len(per_person_verts_raw) >= 2:
            valid_verts = [vl for vl in per_person_verts_raw
                          if any(v is not None for v in vl)]
            if len(valid_verts) >= 2:
                combined_glb_path = os.path.join(
                    args.output_dir, f"{prefix}_combined_mesh.glb"
                )
                print(f"\n    Writing combined mesh GLB ({len(valid_verts)} persons)"
                      f" → {combined_glb_path}")
                write_combined_mesh_glb(
                    combined_glb_path, timestamps, valid_verts, estimator.faces,
                )

    # ── Export OpenSim files ──────────────────────────────────────────────────
    print("\nExporting OpenSim files...")

    # Save metadata JSONs
    inference_time = time.time() - t_start
    meta = {
        "input_video": os.path.abspath(args.video_path),
        "fps": fps,
        "num_frames": total,
        "video_info": {
            "fps": fps,
            "frame_count": total,
            "width": width,
            "height": height,
            "duration": total / fps,
        },
        "inference_time": inference_time,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(outputs_path, "w") as f:
        json.dump(all_raw_outputs, f)

    print(f"  Writing TRC (mm)  → {trc_path}")
    exporter = TRCExporter(fps=out_fps, units="mm")
    exporter.export(markers_array, marker_names, trc_path)

    # --- Optional humanised avatar retarget ------------------------------
    if args.avatar_glb:
        if not os.path.isfile(args.avatar_glb):
            print(f"  WARNING: avatar GLB not found at {args.avatar_glb} — skipping retarget.")
        else:
            avatar_stem = os.path.splitext(os.path.basename(args.avatar_glb))[0]
            # Strip a trailing "_apose"/"_opaque" suffix for nicer output name
            for tag in ("_apose_opaque", "_opaque", "_apose"):
                if avatar_stem.endswith(tag):
                    avatar_stem = avatar_stem[: -len(tag)]
                    break
            avatar_out = os.path.join(
                args.output_dir, f"{prefix}_avatar_{avatar_stem}.glb"
            )
            print(f"  Retargeting → avatar GLB ({avatar_stem})")
            try:
                generate_avatar_from_trc(
                    trc_path=trc_path,
                    avatar_glb_path=args.avatar_glb,
                    out_path=avatar_out,
                )
                print(f"  Avatar GLB        → {avatar_out}")
            except Exception as e:
                print(f"  WARNING: avatar retarget failed: {e}")

    # Scale the generic model to the subject's proportions, then run IK
    if os.path.isfile(model_template):
        shutil.copy(model_template, osim_path)
        print(f"  Writing model     → {osim_path}")
        subject_mass = args.subject_mass

        # Auto-detect the quietest window in the sequence for Scale Tool +
        # MarkerPlacer. Using a static-pose window instead of the whole video
        # avoids averaging over dynamic motion (squat, etc.) — JC distances
        # are stable, MarkerPlacer sees a clean reference pose.
        calib_ts = calib_te = None
        if markerset == "flodelaplace" and getattr(args, "auto_static_calib", True):
            calib_ts, calib_te, f_s, f_e = _detect_static_window(kpts_opensim, out_fps)
            print(f"  [calib] quietest window: frames {f_s}-{f_e}  ({calib_ts:.2f}-{calib_te:.2f}s)")

        # OPT-IN : facteurs d'échelle depuis les JOINTS du rig MHR au lieu des
        # distances entre marqueurs de peau (immunisé à la corpulence).
        # Cf. sam_3d_body/export/mhr_segment_scale.py. Défaut = méthode
        # historique (mesures), inchangée.
        _manual_scales = None
        if getattr(args, "scale_from_mhr_joints", False):
            try:
                from sam_3d_body.export.mhr_segment_scale import (
                    segment_scales_from_joints, subject_joints_from_frames)
                # ⚠️ Utiliser `jcoords_opensim` (sortie de transform()) et NON
                # `all_joint_coords` : ces derniers sont les joints BRUTS de
                # l'inférence, à l'échelle MHR (~11 % compressée). transform()
                # applique `jc = jc * scale` avec le scale dérivé de
                # --person_height → les joints sont alors à la TAILLE RÉELLE,
                # et le ratio vs le template MHR donne directement le bon
                # facteur, sans correction a posteriori.
                _js = subject_joints_from_frames(jcoords_opensim)
                if _js is None:
                    print("  [scale MHR] pas assez de frames avec joints → "
                          "fallback sur les mesures de marqueurs.")
                else:
                    _manual_scales = segment_scales_from_joints(_js)
                    _shown = {k: round(v[0], 4) for k, v in sorted(_manual_scales.items())}
                    print(f"  [scale MHR] facteurs depuis les joints du rig : {_shown}")
            except Exception as _e:
                print(f"  [scale MHR] échec ({_e}) → fallback sur les mesures.")

        print(f"  Scaling model     → {osim_path}  (mass={subject_mass:.1f} kg, height={subject_height:.2f} m)")
        scale_ok = run_scale_tool(
            model_path=osim_path,
            trc_path=trc_path,
            scaled_model_path=osim_path,
            subject_mass=subject_mass,
            subject_height=subject_height,
            markerset=markerset,
            calibration_t_start=calib_ts,
            calibration_t_end=calib_te,
            marker_placer=getattr(args, "marker_placer", False),
            manual_scales=_manual_scales,
        )
        if not scale_ok:
            print("  WARNING: Scale Tool failed – running IK on unscaled model.")
    else:
        print(f"  WARNING: model template not found at {model_template}")

    frames_markers      = [markers_array[i] for i in range(N)]
    frames_markers_body = [markers_body[i]  for i in range(N)]

    print(f"  Running OpenSim IK → {ik_mot_path}")
    ik_ok = run_ik(
        model_path=osim_path,
        trc_path=trc_path,
        mot_path=ik_mot_path,
        errors_path=errors_path,
        markerset=markerset,
    )
    if not ik_ok:
        print("  WARNING: OpenSim IK failed or opensim env not found.")

    # Export TRC post-IK : positions des markers RECALCULÉES depuis le modèle
    # scalé + les angles du .mot. Cohérent à 100 % avec le .mot (mêmes
    # contraintes squelettiques, pas de soft-tissue artifact), contrairement
    # au .trc d'input IK qui contient les positions markers brutes.
    # → consommé par synkro-analytics pour les métriques spatio-temporelles.
    if ik_ok and os.path.isfile(osim_path) and os.path.isfile(ik_mot_path):
        post_ik_trc_path = os.path.join(
            args.output_dir, f"{prefix}_post_ik.trc")
        from sam_3d_body.export.opensim_ik_runner import export_post_ik_trc
        export_post_ik_trc(
            model_path=osim_path,
            mot_path=ik_mot_path,
            output_trc_path=post_ik_trc_path,
        )

    # Per-marker IK error analysis — computes mean/max distance in mm between
    # each TRC marker trajectory and the model's marker FK positions. Useful
    # to spot bony landmarks that fit poorly (bad vertex pick or bad .osim
    # local position). Debug-only opt-in: pass --ik_diagnostics locally when
    # tuning markers; off by default since it roughly doubles the OpenSim
    # step wall time.
    if (ik_ok and args.ik_diagnostics
            and os.path.isfile(osim_path) and os.path.isfile(ik_mot_path)):
        errors_csv = os.path.join(args.output_dir, f"{prefix}_ik_per_marker_errors.csv")
        print(f"  Computing per-marker IK errors → {errors_csv}")
        err_summary = run_per_marker_error_analysis(
            model_path=osim_path, mot_path=ik_mot_path,
            trc_path=trc_path, out_csv=errors_csv,
        )
        if err_summary:
            print(f"\n  Per-marker IK errors (top 15 worst by max, mm):")
            print(f"  {'marker':20s} {'mean':>8s} {'max':>8s}   frames")
            for row in err_summary[:15]:
                print(f"  {row['marker']:20s} {row['mean_mm']:8.2f} {row['max_mm']:8.2f}   {row['n_frames']}")
            # Global summary
            n_total = len(err_summary)
            mean_global = sum(r['mean_mm'] for r in err_summary) / max(n_total, 1)
            max_global = max((r['max_mm'] for r in err_summary), default=0.0)
            print(f"  {'':20s} {'----':>8s} {'----':>8s}")
            print(f"  {f'ALL ({n_total} markers)':20s} {mean_global:8.2f} {max_global:8.2f}")

    # Centre of mass analysis (requires successful IK + scaled model)
    com_ok = False
    if args.compute_com and ik_ok and os.path.isfile(osim_path) and os.path.isfile(ik_mot_path):
        com_path = os.path.join(args.output_dir, f"{prefix}_com.sto")
        print(f"  Computing COM     → {com_path}")
        com_ok = run_com_analysis(osim_path, ik_mot_path, com_path)

    # ── --export_mesh_npz : raw MHR mesh of one frame, estimator camera frame ──
    # Tous les tableaux ci-dessous (all_verts, all_kpts_raw, all_joint_coords)
    # proviennent du même tour de la boucle d'inférence et n'ont subi AUCUNE
    # transformation pipeline (rotation OS, scale, floor lean, etc.) → repère
    # « estimator_camera_raw », unités mètres, cohérence verts/joints/keypoints
    # parfaite pour la frame choisie.
    if args.export_mesh_npz:
        fi = int(args.mesh_npz_frame)
        if fi < 0 or fi >= len(all_verts):
            print(f"  WARNING: --mesh_npz_frame {fi} hors limites "
                  f"(0..{len(all_verts)-1}) — skip export.")
        elif all_verts[fi] is None or all_kpts_raw[fi] is None or all_joint_coords[fi] is None:
            print(f"  WARNING: frame {fi} sans détection (verts/kpts/jcoords None) "
                  f"— skip export.")
        else:
            npz_path = os.path.join(args.output_dir, f"{prefix}_mesh.npz")
            _faces_raw = estimator.faces
            if hasattr(_faces_raw, "detach"):
                _faces_raw = _faces_raw.detach().cpu()
            faces_np = np.asarray(_faces_raw).astype(np.int32)
            np.savez_compressed(
                npz_path,
                verts=np.asarray(all_verts[fi], dtype=np.float32),
                faces=faces_np,
                joint_coords=np.asarray(all_joint_coords[fi], dtype=np.float32),
                keypoints=np.asarray(all_kpts_raw[fi], dtype=np.float32),
                frame_index=np.asarray(fi),
                coordinate_frame=np.asarray("estimator_camera_raw"),
                units=np.asarray("meters"),
                n_vertices=np.asarray(int(all_verts[fi].shape[0])),
                source=np.asarray(os.path.basename(args.video_path)),
            )
            print(f"  Mesh NPZ          → {npz_path}  (frame {fi}, "
                  f"{all_verts[fi].shape[0]} verts, estimator_camera_raw)")

    if not args.no_mesh_glb:
        print(f"  Writing mesh GLB  → {mesh_glb}")
        # Propage la pipeline OpenSim complète (rotation axes + scale + pelvis
        # centering + floor lean correction + ground alignment) aux mesh verts
        # via transformer.apply_pipeline_to_verts(). Le transformer a déjà
        # tourné cette pipeline pour les keypoints (ligne 851), on rejoue les
        # mêmes opérations en utilisant l'état caché pour que le mesh soit
        # au même repère world que l'anatomical GLB.
        # Pour la COHÉRENCE intra-GLB (mesh skin + segments/sphères MHR dessinés
        # dedans), on utilise le MÊME ground_offset_mode ET le MÊME offset
        # numérique pour les 3 inputs.
        #   --floor=True  → constant_from_calib : 1 shift Y unique calculé
        #                   depuis les kpts (référence biomécanique = feet
        #                   markers MHR), réutilisé pour mesh + segments.
        #                   Garantit zéro décalage entre eux.
        #   --floor=False → none : pas de shift Y, position 3D réelle préservée.
        # NB : le pipeline TRC/IK/anatomical utilise toujours `align_to_ground`
        # per_frame de transform() — inchangé.
        # --floor=True  → per_frame : feet à Y=0 chaque frame (subject piedssol)
        # --floor=False → constant_from_calib : shift Y calculé sur les 20
        #                 premières frames (assumées standing) appliqué constant.
        #                 Évite que le mesh soit way below ground quand il y a
        #                 pas de ground alignment per_frame.
        # contact_anchor et --floor produisent tous deux des offsets Y PER-FRAME
        # (stockés dans _last_ground_offsets_m) → le mesh doit les rejouer en
        # mode per_frame, sinon il reste centré au bassin (mesh figé) alors que
        # l'anatomical descend.
        _glb_ground_mode = ("per_frame"
                            if (_apply_floor or args.contact_anchor)
                            else "constant_from_calib")
        # Compute shared calib offset depuis les kpts (= référence biomécanique
        # pieds) pour assurer que mesh + kpts segments + joints partagent le
        # même Y zero dans le GLB final. Utilisé en mode constant_from_calib
        # (no --floor). Pour per_frame (--floor / contact_anchor) l'override est ignoré.
        _shared_offset_m = None
        if not _apply_floor and not args.contact_anchor:
            # Fix Y offset mesh vs anatomical (2026-07-08) : on réutilise
            # EXACTEMENT la valeur calculée par transform() (stockée dans
            # _last_constant_offset_m). Zéro écart mesh/anatomical.
            _shared_offset_m = getattr(transformer, "_last_constant_offset_m", None)
            if _shared_offset_m is not None:
                print(f"  Mesh GLB: shared calib offset Y -= {_shared_offset_m:.3f} m "
                      f"(reuses transform()'s exact value → alignment guaranteed)")
        # Pass RAW points (no cam_t added) so apply_pipeline_to_verts mirrors
        # exactly what transform() did for the canonical kpts_opensim — the
        # only cam_t contribution goes through _last_xz_deltas_m (XZ delta
        # from frame 0). This puts mesh + kpts + jcoords in the same frame
        # as the anatomical GLB (which is driven by kpts_opensim → TRC → IK).
        verts_world = transformer.apply_pipeline_to_verts(
            all_verts, output_units="m",
            ground_offset_mode=_glb_ground_mode,
            override_constant_offset_m=_shared_offset_m)
        kpts_world = transformer.apply_pipeline_to_verts(
            [k.copy() if k is not None else None for k in all_kpts_raw],
            output_units="m",
            ground_offset_mode=_glb_ground_mode,
            override_constant_offset_m=_shared_offset_m)
        jc_world = transformer.apply_pipeline_to_verts(
            [j.copy() if j is not None else None for j in all_joint_coords],
            output_units="m",
            ground_offset_mode=_glb_ground_mode,
            override_constant_offset_m=_shared_offset_m)
        # --feet_anchor : applique le même shift global XZ (en mètres) que
        # celui appliqué aux kpts/markers/jcoords pour que le mesh GLB et
        # l'anatomical/IK restent alignés au sol.
        if feet_anchor_shifts_xz is not None:
            for i, dxz in enumerate(feet_anchor_shifts_xz):
                if verts_world[i] is not None:
                    verts_world[i][:, 0] += dxz[0]
                    verts_world[i][:, 2] += dxz[1]
                if kpts_world[i] is not None:
                    kpts_world[i][:, 0] += dxz[0]
                    kpts_world[i][:, 2] += dxz[1]
                if jc_world[i] is not None:
                    jc_world[i][:, 0] += dxz[0]
                    jc_world[i][:, 2] += dxz[1]
        # Anti-skate : applique le MÊME shift XZ (en mètres) au mesh GLB pour
        # qu'il suive le TRC anti-slidé (sinon le mesh dérape alors que le
        # squelette est ancré).
        if _antiskate_shifts is not None:
            for i in range(len(verts_world)):
                if i >= len(_antiskate_shifts):
                    break
                sx, sz = float(_antiskate_shifts[i][0]), float(_antiskate_shifts[i][1])
                if verts_world[i] is not None:
                    verts_world[i][:, 0] += sx
                    verts_world[i][:, 2] += sz
                if kpts_world[i] is not None:
                    kpts_world[i][:, 0] += sx
                    kpts_world[i][:, 2] += sz
                if jc_world[i] is not None:
                    jc_world[i][:, 0] += sx
                    jc_world[i][:, 2] += sz
        # Le writer attend des verts en frame caméra (il fait son X/Y flip).
        # Nos verts sont DÉJÀ en world OpenSim → on signale verts_in_world=True
        # pour que le writer skip son flip et ne touche pas notre repère.
        write_mesh_glb(mesh_glb, timestamps, verts_world, estimator.faces,
                       frames_kpts=kpts_world,
                       frames_cam_t=[np.zeros(3, dtype=np.float32) if ct is not None else None
                                     for ct in all_cam_t],
                       frames_joint_coords=jc_world,
                       body_only=body_only,
                       verts_in_world=True)

        # gltfpack disabled: incompatible with viewer (KHR_mesh_quantization breaks morph targets)
        # _compress_glb(mesh_glb)

    # NOTE: découplé de --no_mesh_glb (2026-08). Ce bloc ne dépend que du
    # .osim scalé + .mot IK : le désactiver avec le mesh supprimait aussi le
    # GLB anatomique ET les 11 colonnes d'angles cliniques du .mot.
    # Separate anatomical-bone GLB (in OpenSim frame, animated by IK .mot)
    if ik_ok and os.path.isfile(osim_path) and os.path.isfile(ik_mot_path):
        anat_glb = os.path.join(args.output_dir, f"{prefix}_anatomical.glb")
        print(f"  Writing anatomical GLB → {anat_glb}")
        from sam_3d_body.export.opensim_exporter import write_anatomical_glb
        # Alignement anatomical/mesh (fix 2026-07-08) :
        # L'anatomical est natif (via IK sur TRC déjà shifté par transform()).
        # Le mesh est shifté par override_constant_offset_m = _last_constant_offset_m.
        # Ces deux références se retrouvent au MÊME repère si on n'ajoute PAS
        # de y_offset supplémentaire à l'anatomical.
        # L'ancien y_offset = -_shared_offset_m compensait un bug de rotation
        # MoGe (Y-UP conv) qui n'existe plus depuis le fix Y-DOWN.
        # Env var ANAT_Y_OFFSET_M pour override manuel si régression.
        _anat_y_offset = float(os.environ.get("ANAT_Y_OFFSET_M", "0.0"))
        # Le mesh GLB peut avoir recu un offset de mise au sol que le TRC n'a
        # PAS recu : c'est le cas quand transform() n'a rien pu caler (MoGe en
        # echec, aucun mode --floor), si bien que apply_pipeline_to_verts calcule
        # son propre offset de calibration pour le seul mesh. L'anatomical, qui
        # derive du TRC, restait alors a la hauteur brute — mesure sur un bird
        # dog : 33,6 cm d'ecart, squelette sous le sol. On lui applique la MEME
        # translation pour que tout le monde partage un seul repere.
        if _shared_offset_m is None and "ANAT_Y_OFFSET_M" not in os.environ:
            _mesh_applied = getattr(transformer, "_replay_constant_offset_m", None)
            if _mesh_applied:
                _anat_y_offset = -float(_mesh_applied)
                print(f"  Anatomical GLB: Y -= {_mesh_applied:.3f} m "
                      f"(meme offset que le mesh → alignement)")
        write_anatomical_glb(anat_glb, osim_path, ik_mot_path,
                             y_offset_m=_anat_y_offset)

        # Derived clinical angles : ajoute 11 colonnes au .mot
        # (knee_valgus, knee_rotation, ankle_rotation, foot_progression
        # × R/L + trunk_flexion/lean_lateral/rotation). Réutilise le
        # body_transforms.json déjà calculé pour l'anatomical GLB —
        # négligeable en coût supplémentaire (juste de la géométrie).
        from sam_3d_body.export.clinical_angles import add_clinical_angles_to_mot
        body_tf_json = os.path.join(
            args.output_dir,
            f"{os.path.splitext(os.path.basename(osim_path))[0]}_body_transforms.json",
        )
        try:
            n_added = add_clinical_angles_to_mot(ik_mot_path, body_tf_json)
            if n_added:
                print(f"  Clinical angles    → {n_added} colonnes ajoutées au .mot")
        except Exception as err:
            print(f"  [clinical_angles] failed silently: {err}")

    # Write processing report (matches SAM3D-OpenSim convention)
    report_path = os.path.join(args.output_dir, "processing_report.json")
    total_time = time.time() - t_start
    report = {
        "input": os.path.abspath(args.video_path),
        "output_dir": args.output_dir,
        "subject": {"height": subject_height},
        "video_info": {"fps": fps, "frame_count": total, "width": width, "height": height},
        "processing": {
            "fps": out_fps,
            "num_frames": processed,
            "num_markers": len(marker_names),
            "ik_success": ik_ok,
        },
        "timings": {"total": total_time},
        "outputs": {
            "video": vid_path,
            "trc": trc_path,
            "mot": ik_mot_path if ik_ok else None,
            "model": osim_path,
            "mesh_glb": mesh_glb,
            "per_person_trcs": per_person_trcs if len(per_person_trcs) > 0 else None,
            "per_person_ik": per_person_ik_results if len(per_person_ik_results) > 0 else None,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOutput folder: {args.output_dir}")
    print("\nOutput files:")
    print(f"  Video:               {os.path.basename(vid_path)}")
    print(f"  TRC (73 markers/mm): {os.path.basename(trc_path)}")
    print(f"  IK MOT (40 DOF):     {os.path.basename(ik_mot_path)}" + (" ✓" if ik_ok else " (skipped)"))
    print(f"  Body model:          {os.path.basename(osim_path)}")
    if not args.no_mesh_glb:
        print(f"  Mesh GLB:            {os.path.basename(mesh_glb)}")
    print(f"  Processing report:   {os.path.basename(report_path)}")

    print("""
─────────────────────────────────────────────────────────────────
OpenSim workflow:
  1. Load markers_output_<name>_model.osim in OpenSim
  2. Scale Tool → use TRC for static pose calibration
  3. IK Tool → load TRC → IK MOT is already written if opensim env found
  4. GLB files can be previewed in Blender / any glTF viewer (File → Import → glTF 2.0).
─────────────────────────────────────────────────────────────────
""")

    # ─── Auto-analytics via synkro-analytics ─────────────────────────
    # Après SAM3D, si --module fourni, on lance synkro-analytics.cli en
    # subprocess dans l'env opensim (qui a torch + nimble + opensim + reports).
    # Output : metrics.json + report.html + report.pdf dans args.output_dir.
    if getattr(args, "module", None) and args.mass_kg:
        import subprocess as _sp
        opensim_py = os.environ.get("OPENSIM_PYTHON_PATH",
                                     "/opt/conda/envs/opensim/bin/python")
        if not os.path.isfile(opensim_py):
            # Local dev fallback
            for _p in ("/home/fdela/miniconda3/envs/opensim/bin/python",
                       "/home/fdela/miniconda3/envs/fast_sam_3d_body/bin/python"):
                if os.path.isfile(_p):
                    opensim_py = _p
                    break
        # Build subject_info.json
        subj_info = {
            "mass_kg": float(args.mass_kg),
            "height_cm": float(args.person_height) * 100 if args.person_height else 175,
            "level": args.level,
            "osim_path": osim_path,
            "geometry_folder": os.path.join(
                os.path.dirname(osim_path), "Geometry")
            if os.path.isdir(os.path.join(os.path.dirname(osim_path), "Geometry"))
            else None,
            "enable_grf": True,
            "video_path": args.video_path,
        }
        if args.age is not None:
            subj_info["age"] = args.age
        if args.sex is not None:
            subj_info["sex"] = args.sex
        if args.module == "d3.cycling":
            # Position cycliste → stratum de normes (road=performance, tt, comfort).
            # Évite de flaguer à tort coude ~90° / CdA bas en position chrono.
            subj_info["cycling_position"] = args.cycling_position
            subj_info["use_case"] = {"road": "performance", "tt": "tt",
                                     "comfort": "comfort"}[args.cycling_position]
            # Côté caméra (vue 3/4) → neutralise les angles du membre occulté.
            # Vide = auto-détection dans le module (orientation du torse).
            if args.camera_side:
                subj_info["camera_side"] = args.camera_side
        subj_json = os.path.join(args.output_dir, "subject_info.json")
        with open(subj_json, "w") as f:
            json.dump(subj_info, f, indent=2)
        # Call synkro-analytics CLI
        out_dir = os.path.join(args.output_dir, "analytics")
        cli_cmd = [opensim_py, "-m", "synkro_analytics.cli",
                   "--module", args.module,
                   "--mot", ik_mot_path,
                   "--trc", os.path.join(args.output_dir, f"{prefix}_post_ik.trc"),
                   "--subject", subj_json,
                   "--out", out_dir,
                   "--export-pdf"]
        if args.treadmill_speed is not None:
            cli_cmd += ["--treadmill-speed", str(args.treadmill_speed)]
        if getattr(args, "hop_type", None):
            cli_cmd += ["--hop-type", args.hop_type]
        if getattr(args, "leg", None):
            cli_cmd += ["--leg", args.leg]
        print(f"\n[analytics] Running: {' '.join(cli_cmd)}")
        try:
            _sp.run(cli_cmd, check=True)
            print(f"[analytics] ✓ Metrics + report in {out_dir}/")
            # Move GRF GLB to parent output dir (à côté du mesh/anatomical GLB)
            # avec le nom cohérent avec la convention pipeline.
            _grf_glb_src = os.path.join(out_dir, "report_grf.glb")
            if os.path.isfile(_grf_glb_src):
                _grf_glb_dst = os.path.join(args.output_dir,
                                              f"{prefix}_grf.glb")
                import shutil as _sh
                _sh.move(_grf_glb_src, _grf_glb_dst)
                print(f"[analytics] ✓ GRF GLB → {_grf_glb_dst}")
        except _sp.CalledProcessError as e:
            print(f"[analytics] ⚠ Failed (exit {e.returncode}). Skipping.")
        except FileNotFoundError as e:
            print(f"[analytics] ⚠ Python env not found ({opensim_py}). Skipping. {e}")


def build_parser():
    """Construit le parser CLI.

    Extrait de `__main__` pour qu'un worker persistant rejoue EXACTEMENT la
    meme ligne de commande que le CLI. Dupliquer les flags ailleurs serait la
    garantie d'une divergence silencieuse dans six mois.
    """
    parser = argparse.ArgumentParser(description="Fast SAM 3D Body – OpenSim Export")
    parser.add_argument("--video_path", default="./videos/aitor_garden_walk.mp4")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Default: auto-generated as "
                             "output_YYYYMMDD_HHMMSS_<videoname> next to the video.")
    parser.add_argument("--detector", default="yolo_pose")
    parser.add_argument("--detector_model", default="./checkpoints/yolo/yolo11m-pose.engine")
    parser.add_argument("--hand_box_source", default="yolo_pose")
    parser.add_argument("--local_checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    parser.add_argument("--hands", action="store_true",
                        help="(legacy) Equivalent to --inference_type full")
    parser.add_argument("--inference_type", default="body", choices=["full", "body"],
                        help="'body' (default) skips hands for speed; "
                             "'full' adds hand markers for IK and GLB.")
    parser.add_argument("--markerset", default="flodelaplace",
                        choices=["pose2sim", "flodelaplace"],
                        help="Which marker set + model template to use. "
                             "'flodelaplace' (default) = mocap-style markerset using the "
                             "64-marker assets/flodelaplace_mocap.osim, with bony landmarks "
                             "derived from MHR mesh vertex picks + direct kpt/armature sources. "
                             "'pose2sim' = original KeypointConverter "
                             "+ pose2sim_wholebody_model.osim.")
    parser.add_argument("--auto_static_calib", action="store_true", default=True,
                        help="(flodelaplace only) Auto-detect the quietest window in the "
                             "sequence and use it for Scale Tool + MarkerPlacer calibration "
                             "instead of the whole video. Default: on.")
    parser.add_argument("--no_auto_static_calib", action="store_false", dest="auto_static_calib",
                        help="Disable auto-static-calib; use whole-video range for scaling.")
    parser.add_argument("--marker_placer", action="store_true", default=False,
                        help="Enable OpenSim MarkerPlacer: after segment scaling, adjusts "
                             "each marker's local position on its body so it matches the "
                             "TRC at the calibration window. Useful when .osim template "
                             "positions don't match the subject's mesh-picked landmarks. "
                             "Default OFF — diagnose raw placement errors first, then turn "
                             "on to clean up IK residuals.")
    parser.add_argument("--scale_from_mhr_joints", action="store_true", default=False,
                        help="MODE EXPÉRIMENTAL : calcule les facteurs d'échelle du "
                             "ScaleTool depuis les JOINTS du rig MHR (127) au lieu des "
                             "distances entre marqueurs de peau. Les joints du rig ne "
                             "répondent qu'au bloc 'scale' de la base de forme : les "
                             "longueurs d'os sont donc immunisées à la corpulence, alors "
                             "qu'une mesure sur la peau allonge les os d'un sujet "
                             "corpulent. Porté de Mesh2Marker (segment_scale.py). "
                             "Défaut OFF = méthode historique par mesures.")
    parser.add_argument("--target_fps", type=float, default=0,
                        help="Process at this FPS (0=all frames, default=0 = no skipping)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many input frames (0=all)")
    parser.add_argument("--no_mesh_glb", action="store_true",
                        help="Skip full body mesh GLB export (saves ~185 MB for long videos)")
    parser.add_argument("--export_mesh_npz", action="store_true",
                        help="Export raw MHR mesh d'UNE frame en .npz (verts/faces/"
                             "joint_coords/keypoints + meta), repère estimator_camera_raw, "
                             "unités mètres. Off par défaut. Frame choisie via "
                             "--mesh_npz_frame. Force la collecte des verts mesh même si "
                             "--no_mesh_glb est passé.")
    parser.add_argument("--mesh_npz_frame", type=int, default=0,
                        help="Index de la frame à exporter quand --export_mesh_npz est actif "
                             "(défaut 0).")
    parser.add_argument("--no_lean_fix", action="store_true",
                        help="Skip automatic forward-lean correction (manual --lean_angle "
                             "or --lean_ref_frame still apply if set)")
    parser.add_argument("--enable_auto_lean_fix", action="store_true",
                        help="Active la spine lean correction AUTO (off par défaut depuis "
                             "que le biais MHR sur acromions rendait l'anatomical penché ≠ mesh)")
    parser.add_argument("--stationary", action="store_true",
                        help="Disable global XZ translation — keeps the person centred at "
                             "origin with feet fixed to the ground. Use for exercises where "
                             "the subject does not walk (squat, CMJ, deadlift, etc.).")
    parser.add_argument("--lock-vertical", dest="lock_vertical", action="store_true",
                        help="Lock aussi la composante Y (verticale) du pelvis — skip "
                             "l'injection cam_t_Y appliquée sinon en mode --stationary. "
                             "Mesh + anatomical restent ainsi tous deux verrouillés en Y et "
                             "parfaitement alignés (même code path). Utiliser pour rendu "
                             "bikefit indoor / home-trainer où l'oscillation du pédalage "
                             "doit rester invisible.")
    parser.add_argument("--lock_lateral", action="store_true",
                        help="Détecte l'axe d'avance via PCA sur la trajectoire XZ du sujet "
                             "et ne garde QUE la composante longitudinale (avance). Élimine "
                             "le zig-zag latéral dû au bruit monoculaire sur la profondeur. "
                             "Idéal pour running / marche / sprint en ligne droite (l'axe "
                             "peut être mix X-Z selon orientation caméra). Combine avec "
                             "--floor_moge pour un rendu propre 3D.")
    parser.add_argument("--compute_com", action="store_true",
                        help="Compute whole-body centre of mass (COM) trajectory from the "
                             "scaled model and IK motion. Writes a _com.sto file with "
                             "time, com_x, com_y, com_z in metres (OpenSim Y-up frame). "
                             "Requires successful IK.")
    parser.add_argument("--ik_diagnostics", action="store_true",
                        help="Opt-in: run the per-marker IK error analysis (mean/max mm "
                             "distance between each TRC marker and its model-FK position). "
                             "This is a debug tool for catching bad vertex picks. Off by "
                             "default because on CPU-only OpenSim it roughly doubles the IK "
                             "step wall time. Pass this flag locally when tuning markers.")
    parser.add_argument("--floor_moge", action="store_true",
                        help="Estimate floor plane from MoGe depth on the first video frame and use its camera-pitch angle to correct forward lean. Requires MoGe to be available.")
    parser.add_argument("--no_floor_moge", action="store_true",
                        help="Desactive l'estimation du sol MoGe (activee par defaut). "
                             "A utiliser quand aucun sol n'est visible dans la video.")
    parser.add_argument("--contact_anchor", action="store_true",
                        help="Ancrage sol conscient du contact : re-ancre le pied en contact "
                             "soutenu (squat → bassin descend) mais préserve la phase de vol "
                             "(course/saut). Unifie et remplace --floor/défaut. Recommandé pour "
                             "avatars + mouvements libres. (Opt-in en cours de validation.)")
    parser.add_argument("--person_height", type=float, default=None,
                        help="Known person height in metres (e.g. 1.69). Scales all 3D output "
                             "so the skeleton height matches this value. Applied to all persons "
                             "unless --person_heights is set.")
    parser.add_argument("--person_heights", type=str, default=None,
                        help="Comma-separated heights in metres for multi-person mode, "
                             "assigned left-to-right as seen in the video. "
                             "E.g. --person_heights 1.69,1.82  (person on the left=1.69m, "
                             "person on the right=1.82m). Overrides --person_height.")
    parser.add_argument("--subject_mass", type=float, default=70.0,
                        help="Subject mass in kg (default 70.0). Used for model scaling only; "
                             "does not affect kinematics.")
    parser.add_argument("--floor_level", action="store_true",
                        help="(Legacy flag) Per-frame ground alignment is always applied.")
    parser.add_argument("--floor", action="store_true",
                        help="Active le pipeline 'mise au sol + redressement' : kpts/jcoords "
                             "passent par align_to_ground (per-frame, pieds à Y=0) + "
                             "correct_floor_lean (one-shot, redressement caméra). Mesh GLB "
                             "shift Y unique calculé sur les premières frames (calib). "
                             "À utiliser pour les mouvements debout (squat, marche). À OMETTRE "
                             "pour les mouvements non-standing (rameur, couché, suspension) "
                             "→ le mesh et squelette restent dans leur position 3D réelle "
                             "sans forcing au sol.")
    parser.add_argument("--floor_seated", action="store_true",
                        help="Variante de --floor pour les mouvements ASSIS (sit-to-stand, "
                             "tests sur chaise, etc.) : pieds à Y=0 chaque frame, MAIS "
                             "désactive le body-vertical correction qui force midfoot→neck "
                             "vertical (faux quand le sujet est assis). Implique --floor.")
    parser.add_argument("--feet_anchor", action="store_true",
                        help="Shift global per-frame qui verrouille le midpoint des pieds "
                             "(LCAL/RCAL ou LAJC/RAJC) à sa position médiane sur toute la "
                             "vidéo. Translate solidairement mesh + anatomical + kpts + "
                             "markers (XZ uniquement). À utiliser pour les exercices où "
                             "le sujet garde les pieds au sol (5STS, tests sur chaise, "
                             "Lasègue). À NE PAS utiliser si les pieds bougent vraiment "
                             "(marche, course).")
    parser.add_argument("--no_shape_lock", action="store_true",
                        help="Désactive le shape-lock SAM3D (ON par défaut). Par défaut, "
                             "après inférence on agrège (médiane) les paramètres morpho "
                             "shape/scale/expr/body_pose[124:130] sur toutes les frames de "
                             "l'essai, puis on régénère les pred_vertices / pred_joint_coords "
                             "/ pred_keypoints_3d via mhr_head._mhr_forward_core avec ce shape "
                             "locké + la pose conservée per-frame. Résultat : le sujet a une "
                             "morphologie constante sur tout l'essai (seul le mouvement varie). "
                             "Évite que les marqueurs (= vertices indexés) fluctuent à cause "
                             "des ré-estimations frame-à-frame du shape par le réseau. À OMETTRE "
                             "uniquement pour débugger / comparer.")
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length x (pixels). Skips MoGe FOV estimation if set.")
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point x (pixels). Defaults to frame_width/2.")
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--lean_angle", type=float, default=None,
                        help="Manual lean correction angle in degrees. Positive tilts the "
                             "skeleton backward (corrects forward lean). Overrides both "
                             "--floor_moge and automatic spine lean correction. "
                             "Example: --lean_angle 5 corrects 5° of forward lean.")
    parser.add_argument("--lean_ref_frame", type=int, default=None,
                        help="Frame index (0-based among processed frames) where the person "
                             "is known to be standing upright. The spine lean measured on "
                             "this frame is used as the correction for ALL frames. "
                             "Works with --floor_moge: MoGe corrects the floor, then the "
                             "ref frame corrects the residual lean per person. "
                             "Example: --lean_ref_frame 0 (use the first frame as reference)")
    parser.add_argument("--lean_cam_pitch_fix", action="store_true",
                        help="Enable experimental camera-pitch-based lean correction")
    parser.add_argument("--multi_person", action="store_true",
                        help="Enable multi-person export: track detections across frames and "
                             "write per-person TRC files + a combined TRC.")
    parser.add_argument("--tracker", default="botsort",
                        choices=["botsort", "bytetrack", "none"],
                        help="Multi-object tracker to use with --multi_person. "
                             "'botsort' (default) handles re-identification after occlusion; "
                             "'bytetrack' is lighter but no re-ID; "
                             "'none' disables tracking (legacy centroid matching).")
    parser.add_argument("--bbox_thr", type=float, default=0.5,
                        help="Detection bbox confidence threshold passed to the estimator (default: 0.5).")
    parser.add_argument("--nms_thr", type=float, default=0.3,
                        help="NMS threshold passed to the estimator (default: 0.3).")
    parser.add_argument("--bbox_thr_force", type=float, default=0.2,
                        help="Force-detection threshold used by fallback detector when some detections miss pose estimates.")
    parser.add_argument("--force_bboxes", action="store_true",
                        help="Always run per-detection bbox inference (expensive) instead of the default top-down gating/fallback.")
    parser.add_argument("--detect_then_infer", action="store_true",
                        help="Run detector first, then send detected boxes to the pose estimator in chunks (avoids TRT batch-profile issues).")
    parser.add_argument("--inference_batch_cap", type=int, default=4,
                        help="Maximum number of person crops to send to the estimator in one chunk. Set 0 to disable chunking.")
    parser.add_argument("--fallback_lower_bbox", type=float, default=0.05,
                        help="Lower bbox confidence for fallback detector pass.")
    parser.add_argument("--fallback_nms", type=float, default=0.9,
                        help="NMS threshold for fallback detector pass.")
    parser.add_argument("--fallback_iou_thresh", type=float, default=0.5,
                        help="IoU threshold used when merging fallback boxes (not implemented: reserved).")
    parser.add_argument("--max_persons", type=int, default=6,
                        help="Maximum number of person detections to consider per frame (keeps top areas).")
    parser.add_argument("--write_combined_trc", action="store_true",
                        help="Also write the combined TRC used for IK in addition to per-person TRCs.")
    parser.add_argument("--run_ik_per_person", action="store_true",
                        help="Run OpenSim scale + IK for each per-person TRC (slow).")
    parser.add_argument("--avatar_glb", default=None,
                        help="Path to a rigged humanoid GLB (MakeHuman 'Default' rig in "
                             "T- or A-pose). When set, retargets the TRC onto this avatar "
                             "and writes markers_<name>_avatar_<avatar_stem>.glb next to "
                             "the standard outputs. Use the opaque-patched variant "
                             "(assets/avatars/*_opaque.glb) to avoid the MakeHuman BLEND "
                             "alpha default.")
    # ── Auto-analytics (synkro-analytics) ────────────────────────────────
    parser.add_argument("--module", default=None,
                        choices=[None, "d3.running", "d3.gait", "d3.squat",
                                 "d3.jump", "d3.sit_to_stand", "d3.cycling",
                                 "d3.sprint_start", "d3.single_leg_hop",
                                 "d3.single_leg_squat"],
                        help="Après SAM3D, appelle synkro-analytics avec ce module. "
                             "d3.running/gait → GaitDynamics GRF. Autres → Newton-CoM GRF. "
                             "Écrit metrics.json + report.html/pdf dans le dossier output.")
    parser.add_argument("--mass_kg", type=float, default=None,
                        help="Masse sujet (kg) — requis pour --module (GRF + normes).")
    parser.add_argument("--age", type=int, default=None,
                        help="Âge sujet (années) — requis par certains modules (STS).")
    parser.add_argument("--sex", default=None, choices=[None, "M", "F"],
                        help="Sexe biologique — pour normes stratifiées.")
    parser.add_argument("--level", default="trained",
                        choices=["trained", "recreational", "clinical", "elite"],
                        help="Niveau sujet — pour normes stratifiées (défaut trained).")
    parser.add_argument("--treadmill_speed", type=float, default=None,
                        help="Vitesse tapis (m/s) — pour --module d3.running/gait sur tapis. "
                             "Laisser vide pour overground.")
    parser.add_argument("--cycling_position", default="road",
                        choices=["road", "tt", "comfort"],
                        help="Position cycliste (module d3.cycling) — bascule le jeu de "
                             "normes coude/tronc/épaule/aéro. 'road' = course cocottes/drops "
                             "(défaut), 'tt' = contre-la-montre/prolongateurs (coude ~90-105°, "
                             "CdA bas = optimal, pas anormal), 'comfort' = position droite.")
    parser.add_argument("--camera_side", default=None, choices=[None, "R", "L"],
                        help="Côté près de la caméra en vue 3/4 (module d3.cycling). "
                             "Le membre opposé (occulté au bas du pédalier) voit ses angles "
                             "genou/cheville neutralisés pour éviter un faux red flag. "
                             "Laisser vide = auto-détection par l'orientation du torse.")
    parser.add_argument("--no_anti_foot_skate", action="store_true",
                        help="Désactive l'anti-glisse du pied au contact (activé PAR DÉFAUT). "
                             "L'anti-glisse fige la position XZ du pied pendant qu'il touche le "
                             "sol (indépendant de --contact_anchor qui, lui, gère aussi l'ancrage "
                             "vertical du bassin). Toujours utile dès qu'il y a contact au sol.")
    parser.add_argument("--hop_type", default=None,
                        choices=[None, "single", "triple", "crossover", "timed6m"],
                        help="Type de saut pour d3.single_leg_hop (RTS) : single (défaut) / "
                             "triple / crossover / timed6m. Relayé à synkro-analytics.")
    parser.add_argument("--fov_once", type=int, default=0, metavar="N",
                        help="Estimer les intrinsèques caméra UNE fois (médiane de N "
                             "frames échantillonnées) au lieu de relancer MoGe à chaque "
                             "frame. 0 = désactivé (comportement historique). 8 est un bon "
                             "point de départ. Gain mesuré ~10 s sur 285 frames. "
                             "Suppose que la focale ne bouge pas pendant la vidéo : à "
                             "éviter sur une prise de vue avec zoom.")
    parser.add_argument("--leg", default=None, choices=[None, "R", "L"],
                        help="Jambe testée pour les tests unipodaux RTS (single_leg_hop/squat). "
                             "Vide = auto-détection. Une vidéo = une jambe ; LSI agrégé côté app.")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
