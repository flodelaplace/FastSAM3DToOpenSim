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

# ── Rajagopal mocap markerset (--markerset flodelaplace) ────────────────────────
# Weighting philosophy:
#   2.0  bony landmarks (vertex picks + acromion + foot) — most reproducible
#   1.0  joint centers from MHR kpts — minor drift from NN inference
#   1.5  foot markers — important for ground contact
#   0.5  head / hand kpts and armature spine joints — noisier, low priority
MARKER_WEIGHTS_FLODELAPLACE: dict[str, float] = {
    # Head: HTOP/Ears strong constraint, Eye/Nose noisy → low weight.
    "Nose":  0.5, "LEye": 0.3, "REye": 0.3, "LEar": 1.8, "REar": 1.8,
    "HTOP":  2.0,
    # Spine armature (jcoord-direct, soft constraint). c_spine1 removed
    # from the XIPH model (not picked in Mesh2Marker).
    "c_spine0": 0.5, "c_spine2": 0.5, "c_spine3": 0.5,
    "c_neck":   1.0, "c_head":   2.0,
    "RCLAV":    0.5, "LCLAV":    0.5,
    # Torso bony — XIPH (xiphoid process, anterior sternum) added by
    # Mesh2Marker, gives a real anterior landmark for torso_X.
    "C7":   2.0, "XIPH": 1.5,
    "RACR": 2.0, "LACR": 2.0,
    # Upper limb — surface clusters only (no more EJC virtual). Clusters
    # RHTO/RHAP/RHBA/RHFR are placed mid-humerus → soft weight, they help
    # the segment orientation but aren't strong distal constraints.
    "RLEL": 2.0, "RMEL": 2.0, "LLEL": 2.0, "LMEL": 2.0,
    "RHTO": 1.0, "RHAP": 1.0, "RHBA": 1.0, "RHFR": 1.0,
    "LHTO": 1.0, "LHAP": 1.0, "LHBA": 1.0, "LHFR": 1.0,
    "RFAradius": 2.0, "RFAulna": 2.0, "LFAradius": 2.0, "LFAulna": 2.0,
    "RFRM": 1.0, "LFRM": 1.0,
    # Wrist + hand — RWrist_hand/LWrist_hand retirés (redondants avec les
    # deux côtés du poignet RFAradius/RFAulna déjà au-dessus). Mesh2Marker
    # 2026-06.
    "RThumb":      0.5, "RIndex":      0.5, "RPinky":      0.5,
    "LThumb":      0.5, "LIndex":      0.5, "LPinky":      0.5,
    "RIndexTip":   0.5, "RPinkyTip":   0.5,
    "LIndexTip":   0.5, "LPinkyTip":   0.5,
    # Pelvis bony (no more HJC virtual)
    "RASI": 2.0, "LASI": 2.0, "RPSI": 2.0, "LPSI": 2.0,
    # Grand trochanters — ajoutés par Mesh2Marker (sur body femur_r/femur_l).
    # Permettent une mesure femur_Y purement intra-fémur (GTR↔épicondyles
    # fémoraux) au lieu de cross-body ASIS↔épicondyles qui traversait l'articu-
    # lation hip et était biaisé par flexion (Florian, 2026-06).
    "RGTR": 2.0, "LGTR": 2.0,
    # Lower limb — surface clusters only (no more KJC/AJC virtual).
    "RLFC": 2.0, "RMFC": 2.0, "LLFC": 2.0, "LMFC": 2.0,
    "RFLT": 1.0, "RFLB": 1.0, "LFLT": 1.0, "LFLB": 1.0,
    "RLMAL": 2.0, "RMMAL": 2.0, "LLMAL": 2.0, "LMMAL": 2.0,
    "RSHN": 1.0, "RTIB": 1.0, "LSHN": 1.0, "LTIB": 1.0,
    # Foot (bony, high priority for ground contact)
    "RCAL": 1.5, "LCAL": 1.5,
    "RTOE": 1.5, "LTOE": 1.5,
    "RMT5": 1.0, "LMT5": 1.0,
    # Semelle (modèle SOLE, 7 patchs par pied). Présents dans le TRC pour la
    # DÉTECTION DE CONTACT au sol, mais volontairement EXCLUS de l'IK
    # (poids 0) : ce sont des points de peau plantaire, ils écraseraient les
    # repères osseux du pied si on les laissait contraindre la cinématique.
    "SOLE_s1_r": 0.0, "SOLE_s1_l": 0.0,
    "SOLE_s2_r": 0.0, "SOLE_s2_l": 0.0,
    "SOLE_s3_r": 0.0, "SOLE_s3_l": 0.0,
    "SOLE_s4_r": 0.0, "SOLE_s4_l": 0.0,
    "SOLE_s5_r": 0.0, "SOLE_s5_l": 0.0,
    "SOLE_s6_r": 0.0, "SOLE_s6_l": 0.0,
    "SOLE_s7_r": 0.0, "SOLE_s7_l": 0.0,
}

# Scale Tool measurements using Rajagopal marker names — non-uniform per-axis.
# OpenSim gait-body-frame convention: X=anterior, Y=superior, Z=lateral-right.
# Bony landmarks (ASIS/PSIS, LFC/MFC, LMAL/MMAL, LEL/MEL, FAradius/FAulna) are
# preferred for width scaling because they're deterministic vertex picks on
# the mesh, whereas joint centers come from the MHR NN (more noise).
_SCALE_MEASUREMENTS_FLODELAPLACE = [
    # ─────────────────── Pelvis ───────────────────
    # Z (largeur) : ASIS + PSIS pour robustesse (Florian).
    # X (profondeur AP) : ASI↔PSI.
    # Y (hauteur) : pas de mesure verticale intra-pelvis fiable (LASI↔LPSI
    #   presque horizontal). On scale Y uniformément avec X et Z en
    #   réutilisant les MÊMES paires sur l'axe Y → ratio Y = moyenne des
    #   4 ratios largeur/profondeur. Sans ça (pelvis_Y = 1.000), le bassin
    #   est trop court verticalement → tibia n'atteint pas la cheville,
    #   genou trop haut, buste descendu (Florian, 2026-06).
    ("pelvis_Z",    [("LASI", "RASI"), ("LPSI", "RPSI")],                     ["pelvis", "sacrum"],                                                     "Z"),
    ("pelvis_X",    [("RASI", "RPSI"), ("LASI", "LPSI")],                     ["pelvis", "sacrum"],                                                     "X"),
    ("pelvis_Y",    [("LASI", "RASI"), ("LPSI", "RPSI"), ("RASI", "RPSI"), ("LASI", "LPSI")], ["pelvis", "sacrum"],                                     "Y"),

    # ─────────────────── Torso ───────────────────
    # Z (largeur) : acromions + ASIS averaged (compromis silhouette).
    # Y (hauteur COMPLÈTE) : acromion → ASIS, des 2 côtés.
    #   On corrige l'ancienne mesure XIPH↔CLAV qui ne capturait que la
    #   moitié haute du tronc (~25 cm) et sous-scalait le torso (Y=1.04
    #   sur straining_ced). ACR↔ASI ≈ 50 cm = vraie hauteur du tronc.
    # X (profondeur AP) : CLAV-C7 (sup), XIPH-c_spine2 (mid). XIPH antérieur
    #   à c_spine2 postérieur ≈ épaisseur du tronc.
    # torso_Y : CLAV↔ASIS et CLAV↔PSIS (clavicules → crêtes iliaques).
    #   Plus stable que ACR↔ASIS car les acromions bougent davantage avec
    #   la flexion d'épaule et la posture (acromion mal estimé par MHR sur
    #   certains sujets → torso sous-scalé, tête trop basse). Les CLAV sont
    #   plus proches de l'axe vertébral et bougent moins. Validé sur straining
    #   2026-06 (ACR-ASIS donnait 0.95 vs CLAV-ASIS plus cohérent).
    ("torso_Z",     [("LACR", "RACR"), ("LASI", "RASI")],                     ["torso"],                                                                "Z"),
    ("torso_X",     [("RCLAV", "C7"), ("LCLAV", "C7"), ("XIPH", "c_spine2")], ["torso"],                                                                "X"),
    ("torso_Y",     [("RCLAV", "RASI"), ("LCLAV", "LASI"),
                     ("RCLAV", "RPSI"), ("LCLAV", "LPSI")],                   ["torso", "lumbar1", "lumbar2", "lumbar3", "lumbar4", "lumbar5"],         "Y"),

    # ─────────────────── Head ───────────────────
    # Splitté en 3 axes (anciennement uniforme X Y Z). Plus fidèle à la
    # morphologie réelle (peut donner un crâne plus ou moins allongé).
    # Y (hauteur) : HTOP → REar + HTOP → LEar (haut crâne → oreilles).
    # Z (largeur) : REar ↔ LEar (largeur bi-auriculaire).
    # X (profondeur AP) : Nose ↔ c_head (antérieur → centre crâne).
    ("head_Y",      [("HTOP", "REar"), ("HTOP", "LEar")],                     ["head"],                                                                 "Y"),
    ("head_Z",      [("REar", "LEar")],                                        ["head"],                                                                 "Z"),
    ("head_X",      [("Nose", "c_head")],                                      ["head"],                                                                 "X"),

    # ─────────────────── Right lower limb ───────────────────
    # femur_Y : PURE INTRA-FÉMUR — grand trochanter (RGTR) ↔ épicondyles
    #   fémoraux (RLFC/RMFC). Tous les 3 markers sont sur le body femur_r,
    #   donc la mesure n'est PAS biaisée par la flexion hip (contrairement à
    #   l'ancienne mesure ASIS-épicondyles qui traversait l'articulation).
    #   Le grand trochanter est ajouté par Mesh2Marker 2026-06.
    # femur_XZ : LFC ↔ MFC (largeur épicondyles fémoraux).
    # tibia_Y : croise les 2 cotés genou × 2 cotés cheville (4 paires)
    #   + MAL↔CAL pour inclure le talon (sinon tibia sous-scalé).
    # tibia_XZ : LMAL ↔ MMAL (largeur malléoles).
    ("femur_r_Y",   [("RGTR", "RLFC"), ("RGTR", "RMFC")],                     ["femur_r", "patella_r"],                                                 "Y"),
    ("femur_r_XZ",  [("RLFC", "RMFC")],                                       ["femur_r", "patella_r"],                                                 "X Z"),
    # tibia_Y : couvre genou (LFC/MFC) → cheville (LMAL/MMAL) PLUS l'extension
    #   vers le talon (LMAL/MMAL → CAL) — la malléole est ~3-4 cm au-dessus
    #   du sol, ajouter MAL↔CAL compense le tibia_Y précédemment sous-scalé
    #   (1.17× au lieu de 1.22× sur straining_ced).
    ("tibia_r_Y",   [("RLFC", "RLMAL"), ("RLFC", "RMMAL"),
                     ("RMFC", "RLMAL"), ("RMFC", "RMMAL"),
                     ("RLMAL", "RCAL"), ("RMMAL", "RCAL")],                   ["tibia_r"],                                                              "Y"),
    ("tibia_r_XZ",  [("RLMAL", "RMMAL")],                                     ["tibia_r"],                                                              "X Z"),
    ("foot_r",      [("RCAL", "RTOE")],                                       ["talus_r", "calcn_r", "toes_r"],                                         "X Y Z"),

    # ─────────────────── Right upper limb ───────────────────
    # humerus_Y : marqueurs ANATOMIQUES standards — acromion (RACR) ↔
    #   épicondyles huméraux (RLEL/RMEL). Mesure cross-body qui inclut
    #   l'offset acromion → tête humérale, c'est l'approche classique
    #   biomeca clinique. Plus robuste que les clusters bras (RHTO/RHAP/
    #   RHBA/RHFR), qui dépendent du placement Mesh2Marker non standardisé.
    #   En pratique le ratio reste cohérent quand la fenêtre statique est
    #   prise en debout neutre (bras le long du corps, pas d'élévation
    #   glénohumérale). Validé visuellement par Florian (2026-06).
    # humerus_XZ : LEL ↔ MEL (largeur épicondyles huméraux).
    # forearm_Y : couvre les 2 côtés coude × 2 côtés poignet (4 paires).
    # forearm_XZ : FAradius ↔ FAulna (largeur styloïdes).
    ("humerus_r_Y", [("RACR", "RLEL"), ("RACR", "RMEL")],                     ["humerus_r"],                                                            "Y"),
    ("humerus_r_XZ",[("RLEL", "RMEL")],                                       ["humerus_r"],                                                            "X Z"),
    ("forearm_r_Y", [("RLEL", "RFAradius"), ("RLEL", "RFAulna"),
                     ("RMEL", "RFAradius"), ("RMEL", "RFAulna")],             ["ulna_r", "radius_r"],                                                   "Y"),
    ("forearm_r_XZ",[("RFAradius", "RFAulna")],                               ["ulna_r", "radius_r"],                                                   "X Z"),

    # ─────────────────── Left lower limb ───────────────────
    ("femur_l_Y",   [("LGTR", "LLFC"), ("LGTR", "LMFC")],                     ["femur_l", "patella_l"],                                                 "Y"),
    ("femur_l_XZ",  [("LLFC", "LMFC")],                                       ["femur_l", "patella_l"],                                                 "X Z"),
    ("tibia_l_Y",   [("LLFC", "LLMAL"), ("LLFC", "LMMAL"),
                     ("LMFC", "LLMAL"), ("LMFC", "LMMAL"),
                     ("LLMAL", "LCAL"), ("LMMAL", "LCAL")],                   ["tibia_l"],                                                              "Y"),
    ("tibia_l_XZ",  [("LLMAL", "LMMAL")],                                     ["tibia_l"],                                                              "X Z"),
    ("foot_l",      [("LCAL", "LTOE")],                                       ["talus_l", "calcn_l", "toes_l"],                                         "X Y Z"),

    # ─────────────────── Left upper limb ───────────────────
    ("humerus_l_Y", [("LACR", "LLEL"), ("LACR", "LMEL")],                     ["humerus_l"],                                                            "Y"),
    ("humerus_l_XZ",[("LLEL", "LMEL")],                                       ["humerus_l"],                                                            "X Z"),
    ("forearm_l_Y", [("LLEL", "LFAradius"), ("LLEL", "LFAulna"),
                     ("LMEL", "LFAradius"), ("LMEL", "LFAulna")],             ["ulna_l", "radius_l"],                                                   "Y"),
    ("forearm_l_XZ",[("LFAradius", "LFAulna")],                               ["ulna_l", "radius_l"],                                                   "X Z"),

    # ─────────────────── Hands (uniform) ───────────────────
    # hand_r/l : RWrist_hand retiré (redondant avec RFAradius/RFAulna déjà
    # sur radius/ulna). On utilise la largeur de la paume (Index↔Pinky aux
    # MCPs) comme proxy de la taille de main — uniforme XYZ.
    ("hand_r",      [("RIndex", "RPinky")],                                   ["hand_r"],                                                               "X Y Z"),
    ("hand_l",      [("LIndex", "LPinky")],                                   ["hand_l"],                                                               "X Y Z"),
]


# ── Original (pose2sim) markerset ─────────────────────────────────────────────
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
    # Trunk + head: shoulder-to-shoulder width applied uniformly (X Y Z).
    # Previously a separate "torso_height" measurement used c_spine0→Neck for Y, but
    # that pair has a large X-axis offset during walking (body leans forward from camera
    # pitch), inflating the 3D Euclidean distance by ~15–20% and giving an erroneously
    # high Y scale (~1.30× vs expected ~1.02×). The lateral shoulder width measurement
    # is unaffected by forward lean and gives a proportionally correct uniform scale.
    ("torso_width",  [("LShoulder", "RShoulder")],    ["torso", "Abdomen", "lumbar1", "lumbar2", "lumbar3", "lumbar4", "lumbar5", "head"],  "X Y Z"),
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
    # Note: head is included in torso_width (above) to scale uniformly with the torso.
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
    measurements: list | None = None,
    marker_placer: bool = False,
    placer_t_start: float | None = None,
    placer_t_end: float | None = None,
    manual_scales: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    if measurements is None:
        measurements = _SCALE_MEASUREMENTS
    # ScaleTool resolves marker_file and output_model_file relative to the XML
    # file's directory. Use relative paths; keep model_file absolute (loaded
    # from the assets directory, not the output directory).
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    trc_rel          = os.path.relpath(os.path.abspath(trc_path),           xml_dir)
    out_model_rel    = os.path.relpath(os.path.abspath(output_model_path),  xml_dir)
    scale_set_rel    = os.path.relpath(os.path.abspath(scale_set_path),     xml_dir)

    trc_set = set(trc_marker_names)

    meas_xml_parts = []
    for name, pairs, bodies, axes in measurements:
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

    # Mode MANUEL (opt-in) : les facteurs viennent des joints du rig MHR
    # (cf. mhr_segment_scale.py) au lieu des distances entre marqueurs de peau.
    # On bascule le ModelScaler en `manualScale` + ScaleSet explicite ; le
    # MeasurementSet est alors ignoré (on l'écrit vide pour rester lisible).
    if manual_scales:
        scale_objs = "\n".join(
            f'\t\t\t\t\t<Scale name="{body}">\n'
            f'\t\t\t\t\t\t<scales> {s[0]:.6f} {s[1]:.6f} {s[2]:.6f} </scales>\n'
            f'\t\t\t\t\t\t<segment>{body}</segment>\n'
            f'\t\t\t\t\t\t<apply>true</apply>\n'
            f'\t\t\t\t\t</Scale>'
            for body, s in sorted(manual_scales.items())
        )
        scale_set_xml = (
            f'\t\t\t<ScaleSet>\n'
            f'\t\t\t\t<objects>\n'
            f'{scale_objs}\n'
            f'\t\t\t\t</objects>\n'
            f'\t\t\t</ScaleSet>'
        )
        if meas_xml_parts:
            # MODE HYBRIDE : les facteurs manuels (joints du rig MHR, immunisés
            # à la corpulence) couvrent le squelette ; les mesures marqueurs
            # restantes couvrent ce que l'armature ne sait pas mesurer — la
            # TÊTE en premier lieu (aucun span osseux fiable, et elle a besoin
            # de 3 axes indépendants). Les deux mécanismes s'appliquent dans
            # l'ordre listé ; ils portent sur des corps disjoints.
            scaler_block = (
                f'\t\t\t<scaling_order> measurements manualScale </scaling_order>\n'
                f'\t\t\t<MeasurementSet>\n'
                f'\t\t\t\t<objects>\n'
                f'{meas_str}\n'
                f'\t\t\t\t</objects>\n'
                f'\t\t\t</MeasurementSet>\n'
                f'{scale_set_xml}'
            )
        else:
            scaler_block = (
                f'\t\t\t<scaling_order> manualScale </scaling_order>\n'
                f'{scale_set_xml}'
            )
    else:
        scaler_block = (
            f'\t\t\t<scaling_order> measurements </scaling_order>\n'
            f'\t\t\t<MeasurementSet>\n'
            f'\t\t\t\t<objects>\n'
            f'{meas_str}\n'
            f'\t\t\t\t</objects>\n'
            f'\t\t\t</MeasurementSet>'
        )

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
{scaler_block}
\t\t\t<marker_file>{trc_rel}</marker_file>
\t\t\t<time_range>{t_start:.6f} {t_end:.6f}</time_range>
\t\t\t<output_model_file>{out_model_rel}</output_model_file>
\t\t\t<output_scale_file>{scale_set_rel}</output_scale_file>
\t\t\t<preserve_mass_distribution>true</preserve_mass_distribution>
\t\t</ModelScaler>
{_make_marker_placer_xml(marker_placer, trc_rel, out_model_rel,
                         placer_t_start if placer_t_start is not None else t_start,
                         placer_t_end   if placer_t_end   is not None else t_end)}
\t</ScaleTool>
</OpenSimDocument>
"""
    Path(xml_path).write_text(xml, encoding="utf-8")


def _make_marker_placer_xml(enable: bool, trc_rel: str, out_model_rel: str,
                            t_start: float, t_end: float) -> str:
    """MarkerPlacer block: after segment scaling, moves every marker on the
    scaled model so that its world position at the calibration pose matches
    the TRC. Removes the constant per-marker offset that would otherwise
    contaminate IK for all frames."""
    if not enable:
        return ('\t\t<MarkerPlacer>\n'
                '\t\t\t<apply>false</apply>\n'
                '\t\t\t<output_model_file></output_model_file>\n'
                '\t\t</MarkerPlacer>')
    return (f'\t\t<MarkerPlacer>\n'
            f'\t\t\t<apply>true</apply>\n'
            f'\t\t\t<marker_file>{trc_rel}</marker_file>\n'
            f'\t\t\t<time_range>{t_start:.6f} {t_end:.6f}</time_range>\n'
            f'\t\t\t<output_model_file>{out_model_rel}</output_model_file>\n'
            f'\t\t</MarkerPlacer>')


# ---------------------------------------------------------------------------
# Inline Scale Tool script executed inside the opensim env
# ---------------------------------------------------------------------------
_SCALE_SCRIPT = """
import sys, json, os, xml.etree.ElementTree as ET
setup_xml        = sys.argv[1]
result_json      = sys.argv[2]
scaled_model_path = sys.argv[3]
scale_set_path   = sys.argv[4]
# argv[5] = "1" if MarkerPlacer was used (then we skip the template-reset
# post-proc so its adjustments aren't overwritten).
skip_marker_reset = (len(sys.argv) > 5 and sys.argv[5] == "1")

try:
    import opensim
    opensim.Logger.setLevelString('error')

    # Snapshot marker positions BEFORE ScaleTool -- these are the template (pre-scale)
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

    if scale_factors and not skip_marker_reset:
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
    markerset: str = "pose2sim",
    calibration_t_start: float | None = None,
    calibration_t_end:   float | None = None,
    marker_placer: bool = False,
    manual_scales: dict[str, tuple[float, float, float]] | None = None,
) -> bool:
    """
    Scale the generic OpenSim model to the subject's proportions using the TRC.

    Uses all available frames for robust average segment-length estimation.
    Returns True on success, False if the opensim env is unavailable or fails.
    """
    opensim_python = _find_opensim_python()
    if opensim_python is None:
        return False

    full_t_start, full_t_end = _get_trc_time_range(trc_path)
    # ModelScaler + MarkerPlacer both use the calibration window if provided,
    # else they fall back to the full TRC range.
    t_start = calibration_t_start if calibration_t_start is not None else full_t_start
    t_end   = calibration_t_end   if calibration_t_end   is not None else full_t_end
    trc_marker_names = _read_trc_marker_names(trc_path)
    output_dir = str(Path(scaled_model_path).parent.resolve())

    with tempfile.TemporaryDirectory() as tmp:
        xml_path       = os.path.join(output_dir, "_scale_setup.xml")
        scale_set_path = os.path.join(output_dir, "_scale_factors.xml")
        result_json    = os.path.join(tmp, "result.json")
        script_path    = os.path.join(tmp, "run_scale.py")

        measurements = (_SCALE_MEASUREMENTS_FLODELAPLACE
                        if markerset == "flodelaplace" else _SCALE_MEASUREMENTS)
        if manual_scales:
            # HYBRIDE : on ne garde que les mesures portant sur des corps que
            # les facteurs MHR ne couvrent PAS (la tête n'a pas de span
            # d'armature fiable). Évite que les deux mécanismes se marchent
            # dessus sur un même corps.
            _covered = set(manual_scales)
            measurements = [m for m in measurements
                            if not (set(m[2]) & _covered)]
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
            measurements=measurements,
            marker_placer=marker_placer,
            placer_t_start=t_start,
            placer_t_end=t_end,
            manual_scales=manual_scales,
        )
        Path(script_path).write_text(_SCALE_SCRIPT, encoding="utf-8")

        result = subprocess.run(
            [opensim_python, script_path, xml_path, result_json,
             os.path.abspath(scaled_model_path), os.path.abspath(scale_set_path),
             "1" if marker_placer else "0"],
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


# Fallback paths when OPENSIM_PYTHON_PATH env var is not set
_OPENSIM_PYTHON_CANDIDATES = [
    "/opt/conda/envs/opensim/bin/python",           # Docker
    os.path.expanduser("~/miniconda3/envs/opensim/bin/python"),  # Local
]


def _find_opensim_python() -> str | None:
    """Find the opensim conda Python interpreter.

    Checks OPENSIM_PYTHON_PATH env var first, then known fallback paths.
    """
    env_path = os.environ.get("OPENSIM_PYTHON_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
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
    weights: dict[str, float] | None = None,
) -> None:
    if weights is None:
        weights = MARKER_WEIGHTS
    trc_set = set(trc_marker_names)
    tasks_xml = []
    for name, weight in weights.items():
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
    Path(xml_path).write_text(xml, encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Inline post-IK TRC script — recompute marker positions FROM the IK'd state
# (each marker's getLocationInGround(state) after applying the .mot angles).
# Output : un .trc avec les positions "vues" par le modèle, cohérentes à 100 %
# avec le .mot (contraintes squelettiques respectées, soft-tissue artifact
# absent, contrairement au .trc d'input qui contient les positions markers
# brutes). En unités millimètres (convention OpenSim TRC).
# ---------------------------------------------------------------------------
_POST_IK_TRC_SCRIPT = """
import sys, json, os, math
model_path  = sys.argv[1]
mot_path    = sys.argv[2]
trc_path    = sys.argv[3]
result_json = sys.argv[4]

try:
    import opensim
    opensim.Logger.setLevelString('error')

    model = opensim.Model(model_path)
    state = model.initSystem()

    # Charge le .mot
    table = opensim.TimeSeriesTable(mot_path)
    times = list(table.getIndependentColumn())
    col_labels = list(table.getColumnLabels())
    n_frames = len(times)

    # Récupère les coordonnées du modèle et matche avec les colonnes du .mot
    # Les coords translation du pelvis (pelvis_tx/ty/tz) sont en mètres dans
    # le .mot ; toutes les autres sont en degrés. Inutile d'utiliser
    # getMotionType() (renommé/inconsistant entre versions OpenSim).
    _TRANS_NAMES = {'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}
    coord_set = model.getCoordinateSet()
    n_coords = coord_set.getSize()
    coord_info = []  # (coord_obj, col_index_in_mot, is_rotational)
    for ci in range(n_coords):
        c = coord_set.get(ci)
        nm = c.getName()
        if nm in col_labels:
            col_idx = col_labels.index(nm)
            is_rot = nm not in _TRANS_NAMES
            coord_info.append((c, col_idx, is_rot))

    # Liste des markers
    markers = model.getMarkerSet()
    n_markers = markers.getSize()
    marker_names = [markers.get(mi).getName() for mi in range(n_markers)]

    # Pour chaque frame, set coord values, realize, lire positions markers
    # positions[fi][mi] = (x,y,z) en mètres
    positions = []
    for fi in range(n_frames):
        row = table.getRowAtIndex(fi)
        for c, col_idx, is_rot in coord_info:
            val = row[col_idx]
            if is_rot:
                val = val * math.pi / 180.0   # deg → rad
            c.setValue(state, val, False)  # no assemble
        model.assemble(state)
        model.realizePosition(state)
        frame_pos = []
        for mi in range(n_markers):
            pos = markers.get(mi).getLocationInGround(state)
            frame_pos.append((pos[0], pos[1], pos[2]))
        positions.append(frame_pos)

    # Écrit le TRC (en mm, convention OpenSim)
    fps = (n_frames - 1) / (times[-1] - times[0]) if n_frames > 1 else 60.0
    lines = []
    lines.append(f"PathFileType\\t4\\t(X/Y/Z)\\t{os.path.basename(trc_path)}")
    lines.append("DataRate\\tCameraRate\\tNumFrames\\tNumMarkers\\tUnits\\tOrigDataRate\\tOrigDataStartFrame\\tOrigNumFrames")
    lines.append(f"{fps:.6f}\\t{fps:.6f}\\t{n_frames}\\t{n_markers}\\tmm\\t{fps:.6f}\\t1\\t{n_frames}")
    header = ["Frame#", "Time"]
    for nm in marker_names:
        header += [nm, "", ""]
    lines.append("\\t".join(header))
    sub = ["", ""]
    for k in range(n_markers):
        sub += [f"X{k+1}", f"Y{k+1}", f"Z{k+1}"]
    lines.append("\\t".join(sub))
    lines.append("")  # blank line
    for fi in range(n_frames):
        row = [str(fi + 1), f"{times[fi]:.6f}"]
        for x, y, z in positions[fi]:
            row += [f"{x*1000:.4f}", f"{y*1000:.4f}", f"{z*1000:.4f}"]
        lines.append("\\t".join(row))

    with open(trc_path, "w") as f:
        f.write("\\n".join(lines) + "\\n")

    json.dump({"ok": True, "n_frames": n_frames, "n_markers": n_markers}, open(result_json, "w"))
except Exception as e:
    import traceback
    json.dump({"ok": False, "error": str(e), "trace": traceback.format_exc()}, open(result_json, "w"))
    sys.exit(1)
"""


def export_post_ik_trc(
    model_path: str,
    mot_path: str,
    output_trc_path: str,
) -> bool:
    """Recompute marker positions from the IK'd model state, write a .trc.

    Pour chaque timestep du `.mot`, applique les angles articulaires à l'état
    du modèle scalé, puis lit la position de chaque marker dans le ground
    frame. Le TRC produit est cohérent à 100 % avec le `.mot` (mêmes contraintes
    squelettiques), à la différence du `.trc` d'input IK qui contient les
    positions markers brutes (peau, soft-tissue artifact présent).

    Returns True on success, False if opensim env unavailable or script fails.
    """
    opensim_python = _find_opensim_python()
    if opensim_python is None:
        print("  [post-IK TRC] opensim conda env not found – skipping")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        script_path = os.path.join(tmp, "post_ik_trc.py")
        result_json = os.path.join(tmp, "result.json")
        Path(script_path).write_text(_POST_IK_TRC_SCRIPT, encoding="utf-8")

        result = subprocess.run(
            [opensim_python, script_path,
             os.path.abspath(model_path),
             os.path.abspath(mot_path),
             os.path.abspath(output_trc_path),
             result_json],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [post-IK TRC] failed (returncode={result.returncode}):")
            print(f"    stderr: {result.stderr[:500]}")
            if os.path.exists(result_json):
                r = json.load(open(result_json))
                print(f"    error: {r.get('error')}")
                print(f"    trace: {r.get('trace', '')[:500]}")
            return False
        if os.path.exists(result_json):
            r = json.load(open(result_json))
            if not r.get("ok"):
                print(f"  [post-IK TRC] error: {r.get('error')}")
                return False
            print(f"  [post-IK TRC] wrote {r.get('n_markers')} markers × "
                  f"{r.get('n_frames')} frames → {os.path.basename(output_trc_path)}")
    return True


def run_ik(
    model_path: str,
    trc_path: str,
    mot_path: str,
    errors_path: str | None = None,
    markerset: str = "pose2sim",
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

        weights = (MARKER_WEIGHTS_FLODELAPLACE
                   if markerset == "flodelaplace" else MARKER_WEIGHTS)
        _write_ik_setup_xml(
            model_path=os.path.abspath(model_path),
            trc_path=os.path.abspath(trc_path),
            mot_path=os.path.abspath(mot_path),
            time_start=time_start,
            time_end=time_end,
            trc_marker_names=trc_marker_names,
            xml_path=xml_path,
            weights=weights,
        )
        Path(script_path).write_text(_IK_SCRIPT, encoding="utf-8")

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


# ---------------------------------------------------------------------------
# Per-marker IK error analysis — re-runs FK on the scaled model at each IK
# frame and compares model-frame marker positions to the TRC positions.
# Gives you a per-marker mean/max error table for spotting bad picks.
# ---------------------------------------------------------------------------
_PER_MARKER_ERRORS_SCRIPT = """
import sys, json, math, os

osim_path   = sys.argv[1]
mot_path    = sys.argv[2]
trc_path    = sys.argv[3]
out_csv     = sys.argv[4]
result_json = sys.argv[5]

def parse_trc(path):
    '''Minimal TRC parser: returns (times_list, {marker_name: [(x,y,z),...]})
    in meters regardless of the file unit (mm or m).'''
    with open(path) as f:
        lines = f.read().splitlines()
    # Line 0: "PathFileType ...", line 1: metadata header, line 2: metadata values,
    # line 3: "Frame#\\tTime\\tMarker1\\t\\t\\tMarker2...", line 4: coord labels
    # line 5: blank, line 6+: data rows.
    meta_vals = lines[2].split('\\t')
    units = meta_vals[4].strip() if len(meta_vals) > 4 else 'mm'
    scale = 0.001 if units == 'mm' else 1.0
    hdr = lines[3].split('\\t')
    marker_names = [h.strip() for h in hdr[2:] if h.strip()]
    # Skip to data (first non-blank line after line 4).
    data_start = 5
    while data_start < len(lines) and not lines[data_start].strip():
        data_start += 1
    times = []
    m_data = {n: [] for n in marker_names}
    for ln in lines[data_start:]:
        parts = ln.split('\\t')
        if len(parts) < 2:
            continue
        try:
            t = float(parts[1])
        except ValueError:
            continue
        times.append(t)
        for k, name in enumerate(marker_names):
            col = 2 + k * 3
            try:
                x = float(parts[col])     * scale
                y = float(parts[col + 1]) * scale
                z = float(parts[col + 2]) * scale
            except (IndexError, ValueError):
                x = y = z = float('nan')
            m_data[name].append((x, y, z))
    return times, m_data

try:
    import opensim
    opensim.Logger.setLevelString('error')

    trc_times, trc_data = parse_trc(trc_path)
    model = opensim.Model(osim_path)
    state = model.initSystem()

    storage = opensim.Storage(mot_path)
    col_labels = storage.getColumnLabels()

    _TRANS_NAMES = {'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}
    coord_set = model.getCoordinateSet()
    coord_map = {}
    for i in range(coord_set.getSize()):
        c = coord_set.get(i)
        name = c.getName()
        idx = col_labels.findIndex(name)
        if idx >= 0:
            coord_map[name] = (c, idx, name not in _TRANS_NAMES)

    marker_set = model.getMarkerSet()
    model_marker_names = [marker_set.get(i).getName() for i in range(marker_set.getSize())]
    tracked = [m for m in model_marker_names if m in trc_data]

    err_sum    = {m: 0.0 for m in tracked}
    err_max    = {m: 0.0 for m in tracked}
    err_count  = {m: 0   for m in tracked}

    n_frames = storage.getSize()
    tlist = trc_times
    for i in range(n_frames):
        sv = storage.getStateVector(i)
        t  = sv.getTime()
        data = sv.getData()
        for name, (coord, col_idx, is_rot) in coord_map.items():
            val = float(data.get(col_idx - 1))
            if is_rot:
                val = math.radians(val)
            coord.setValue(state, val)
        model.realizePosition(state)
        # Nearest TRC frame (assume monotone times; binary search not needed).
        # fps match between mot and trc → index == i in the common case.
        if i < len(tlist):
            j = i
        else:
            j = min(range(len(tlist)), key=lambda k: abs(tlist[k] - t))
        for name in tracked:
            mk = marker_set.get(name)
            pm = mk.getLocationInGround(state)
            pt = trc_data[name][j]
            if any(v != v for v in pt):  # NaN check
                continue
            dx = pm[0] - pt[0]; dy = pm[1] - pt[1]; dz = pm[2] - pt[2]
            e  = math.sqrt(dx*dx + dy*dy + dz*dz)
            err_sum[name]   += e
            err_max[name]    = max(err_max[name], e)
            err_count[name] += 1

    with open(out_csv, 'w') as f:
        f.write('marker,mean_mm,max_mm,n_frames\\n')
        for name in tracked:
            n  = err_count[name] or 1
            m_ = err_sum[name] * 1000.0 / n
            mx = err_max[name] * 1000.0
            f.write(f'{name},{m_:.2f},{mx:.2f},{err_count[name]}\\n')

    summary = [{'marker': n,
                'mean_mm': err_sum[n]*1000.0/(err_count[n] or 1),
                'max_mm':  err_max[n]*1000.0,
                'n_frames': err_count[n]} for n in tracked]
    json.dump({"ok": True, "summary": summary, "n_frames": n_frames},
              open(result_json, 'w'))
except Exception as e:
    import traceback
    json.dump({"ok": False, "error": str(e), "traceback": traceback.format_exc()},
              open(result_json, 'w'))
    sys.exit(1)
"""


def run_per_marker_error_analysis(
    model_path: str,
    mot_path: str,
    trc_path: str,
    out_csv: str,
) -> list | None:
    """Compute per-marker mean/max distance between TRC positions and the IK
    model's marker positions (both in meters → reported in millimeters).

    Returns a list of dicts {marker, mean_mm, max_mm, n_frames} sorted by
    descending max_mm, or None on failure. Also writes a CSV to *out_csv*.
    """
    opensim_python = _find_opensim_python()
    if opensim_python is None:
        print("  [IK errors] opensim conda env not found – skipping per-marker analysis")
        return None
    for p in (model_path, mot_path, trc_path):
        if not os.path.isfile(p):
            print(f"  [IK errors] missing input: {p}")
            return None

    with tempfile.TemporaryDirectory() as tmp:
        result_json = os.path.join(tmp, "result.json")
        script_path = os.path.join(tmp, "run_per_marker.py")
        Path(script_path).write_text(_PER_MARKER_ERRORS_SCRIPT, encoding="utf-8")
        child_env = {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
                     "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [opensim_python, script_path,
             os.path.abspath(model_path),
             os.path.abspath(mot_path),
             os.path.abspath(trc_path),
             os.path.abspath(out_csv),
             result_json],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env=child_env,
        )
        if not os.path.exists(result_json):
            msg = (result.stderr or result.stdout or "(no output)")[-500:]
            print(f"  [IK errors] subprocess failed:\n{msg}")
            return None
        r = json.load(open(result_json))
        if not r.get("ok"):
            print(f"  [IK errors] error: {r.get('error')}")
            return None
        summary = sorted(r["summary"], key=lambda s: -s["max_mm"])
        return summary


# ---------------------------------------------------------------------------
# COM analysis — compute whole-body centre of mass from scaled model + IK motion
# ---------------------------------------------------------------------------
_COM_SCRIPT = """
import sys, json, math, os

osim_path   = sys.argv[1]
mot_path    = sys.argv[2]
out_sto     = sys.argv[3]
result_json = sys.argv[4]

try:
    import opensim
    opensim.Logger.setLevelString('error')

    model = opensim.Model(osim_path)
    state = model.initSystem()

    storage = opensim.Storage(mot_path)
    col_labels = storage.getColumnLabels()

    # Build mapping: coordinate name -> column index in storage
    # Pelvis translations (pelvis_tx/ty/tz) are in metres; all others in degrees.
    _TRANS_NAMES = {'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}
    coord_set = model.getCoordinateSet()
    coord_map = {}  # coord_name -> (coord_obj, col_idx, is_rotational)
    for i in range(coord_set.getSize()):
        c = coord_set.get(i)
        name = c.getName()
        idx = col_labels.findIndex(name)
        if idx >= 0:
            coord_map[name] = (c, idx, name not in _TRANS_NAMES)

    times, com_x, com_y, com_z = [], [], [], []

    for i in range(storage.getSize()):
        sv = storage.getStateVector(i)
        t = sv.getTime()
        data = sv.getData()

        for name, (coord, col_idx, is_rot) in coord_map.items():
            val = float(data.get(col_idx - 1))  # -1 because col 0 is time
            if is_rot:
                val = math.radians(val)  # MOT stores degrees for rotational DOFs
            coord.setValue(state, val)

        model.realizePosition(state)
        com = model.calcMassCenterPosition(state)
        times.append(t)
        com_x.append(com[0])
        com_y.append(com[1])
        com_z.append(com[2])

    # Write as .sto (OpenSim storage format)
    with open(out_sto, 'w') as f:
        f.write("Center of Mass\\n")
        f.write("version=1\\n")
        f.write("nRows=%d\\n" % len(times))
        f.write("nColumns=4\\n")
        f.write("inDegrees=no\\n")
        f.write("endheader\\n")
        f.write("time\\tcom_x\\tcom_y\\tcom_z\\n")
        for t, x, y, z in zip(times, com_x, com_y, com_z):
            f.write("%.6f\\t%.6f\\t%.6f\\t%.6f\\n" % (t, x, y, z))

    json.dump({"ok": True, "n_frames": len(times)}, open(result_json, "w"))
except Exception as e:
    import traceback
    json.dump({"ok": False, "error": str(e), "traceback": traceback.format_exc()},
              open(result_json, "w"))
    sys.exit(1)
"""


def run_com_analysis(
    model_path: str,
    mot_path: str,
    com_path: str,
) -> bool:
    """Compute whole-body centre of mass from scaled model + IK motion.

    Writes a .sto file with columns: time, com_x, com_y, com_z (metres,
    OpenSim Y-up frame).  Returns True on success.
    """
    opensim_python = _find_opensim_python()
    if opensim_python is None:
        print("  [COM] opensim conda env not found – skipping COM")
        return False

    if not os.path.isfile(model_path) or not os.path.isfile(mot_path):
        print("  [COM] model or mot file not found – skipping COM")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        result_json = os.path.join(tmp, "result.json")
        script_path = os.path.join(tmp, "run_com.py")
        Path(script_path).write_text(_COM_SCRIPT, encoding="utf-8")

        child_env = {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
                     "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [opensim_python, script_path,
             os.path.abspath(model_path),
             os.path.abspath(mot_path),
             os.path.abspath(com_path),
             result_json],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env=child_env,
        )

        # Read result_json BEFORE leaving the tmpdir context manager
        ok = False
        if os.path.exists(result_json):
            r = json.load(open(result_json))
            if r.get("ok"):
                ok = True
                print(f"  [COM] {r.get('n_frames', '?')} frames → {com_path}")
            else:
                print(f"  [COM] error: {r.get('error', '?')}")
                tb = r.get("traceback", "")
                if tb:
                    print(f"  [COM] traceback:\n{tb[-800:]}")
        elif result.returncode != 0:
            msg = (result.stderr or result.stdout or "(no output)")[-500:]
            print(f"  [COM] subprocess failed (rc={result.returncode}):\n{msg}")

    return ok


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