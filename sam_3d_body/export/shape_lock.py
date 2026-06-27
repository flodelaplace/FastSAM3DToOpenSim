"""Shape-lock pour FastSAM3D body inference.

Porté depuis Mesh2Sim/stages/frontend_mhr/.../shape_lock.py.

Idée : SAM3D body ré-estime la morphologie (shape/scale/expr) à chaque frame
indépendamment. Sur un essai complet, ces estimations fluctuent → les positions
des marqueurs (= vertices indexés) bougent au-delà de la pose réelle, ce qui
brouille le scaling OpenSim (mesures distance entre markers traversant un sujet
"morphologiquement instable") et fait "chasser" l'IK derrière le mesh.

Le shape lock :
1. Médiane des paramètres morphologiques sur toutes les frames valides de
   l'essai → shape locké (= morphologie unique pour le sujet).
2. Re-exécute le rig MHR avec ce shape locké + chaque pose conservée
   frame-par-frame → meshes régénérés où SEUL le mouvement varie.

Params shape (lockés à la médiane) :
  - shape_params (45)    : identity coeffs
  - scale_params (28)    : PCA-encoded segment scales (→ 68 scales via PCA)
  - expr_params (72)     : face expression (no body marker, locké pour repro)
  - body_pose_params[124:130] : 6 shape modes PCA-encoded comme 1-DoF
    translations cachées dans body_pose_params (cf. Mesh2Sim CLAUDE.md
    "Paramètres MHR et chaîne de scaling").

Params pose (per-frame, conservés) :
  - global_rot (3), pred_cam_t (3), body_pose_params (130 pose values),
    hand_pose_params (108).
"""

from __future__ import annotations

import numpy as np
import torch

# Shape modes inside body_pose_params, positions 124..129.
_BODY_POSE_SHAPE_MODE_SLICE = slice(124, 130)


def aggregate_shape(
    shape_params_list,
    scale_params_list,
    expr_params_list,
    body_pose_params_list,
):
    """Médiane par-élément des paramètres morpho sur les frames valides.

    Args:
        *_list: listes de longueur N (= nombre de frames), chaque entrée est
            un np.ndarray ou None (frame ratée).

    Returns:
        dict {shape_params, scale_params, expr_params, body_pose_shape_modes}
        (np.float32).

    Raises:
        ValueError: si aucune frame valide.
    """
    def _median(lst, label):
        valid = [np.asarray(x, dtype=np.float32) for x in lst if x is not None]
        if not valid:
            raise ValueError(f"shape_lock: aucune frame valide pour {label}")
        return np.median(np.stack(valid, axis=0), axis=0).astype(np.float32)

    locked_shape = _median(shape_params_list, "shape_params")
    locked_scale = _median(scale_params_list, "scale_params")
    locked_expr = _median(expr_params_list, "expr_params")

    valid_bp = [np.asarray(x, dtype=np.float32) for x in body_pose_params_list if x is not None]
    if not valid_bp:
        raise ValueError("shape_lock: aucune frame valide pour body_pose_params")
    bp_modes = np.stack(
        [bp[_BODY_POSE_SHAPE_MODE_SLICE] for bp in valid_bp], axis=0
    )
    locked_bp_modes = np.median(bp_modes, axis=0).astype(np.float32)

    return {
        "shape_params": locked_shape,
        "scale_params": locked_scale,
        "expr_params": locked_expr,
        "body_pose_shape_modes": locked_bp_modes,
    }


def regenerate_with_locked_shape(
    mhr_head,
    locked,
    per_frame_pose,
    device="cuda",
):
    """Re-exécute le rig MHR avec shape locké pour chaque frame.

    Args:
        mhr_head: instance MHRHead (= estimator.model.head_pose).
        locked: dict produit par :func:`aggregate_shape`.
        per_frame_pose: liste (longueur N) de dicts ou None. Chaque dict
            contient :
              - global_trans (3,)   = pred_cam_t en mètres
              - global_rot (3,)
              - body_pose_params    (>= 130,)
              - hand_pose_params    (108,) ou None
        device: torch device ("cuda" recommandé).

    Returns:
        liste de longueur N. Chaque entrée :
          - None si l'input était None,
          - sinon dict {pred_vertices (V,3), pred_joint_coords (J,3),
            pred_keypoints_3d (K,3)} en repère MHR (cam-relative, mètres).
        Garde la sémantique de `process_one_image` : `pred_vertices` reste
        relatif (le pipeline appliquera ensuite cam_t et la coord transform).
    """
    dev = torch.device(device)

    # Tensors lockés (broadcast batch=1).
    t_shape = torch.from_numpy(locked["shape_params"]).to(dev).unsqueeze(0)   # (1, 45)
    t_scale = torch.from_numpy(locked["scale_params"]).to(dev).unsqueeze(0)   # (1, 28)
    t_expr = torch.from_numpy(locked["expr_params"]).to(dev).unsqueeze(0)     # (1, 72)
    locked_bp = locked["body_pose_shape_modes"].astype(np.float32)            # (6,)

    results = []
    mhr_head.eval()
    with torch.no_grad():
        for pose in per_frame_pose:
            if pose is None:
                results.append(None)
                continue

            # global_trans = 0 : le rig sort en repère cam-relative pur. Le
            # pipeline ajoute cam_t plus tard (idem que `process_one_image`
            # qui appelle mhr_forward avec global_trans = global_rot * 0).
            gt = torch.zeros(1, 3, device=dev, dtype=torch.float32)
            gr = torch.from_numpy(
                np.asarray(pose["global_rot"], dtype=np.float32)
            ).to(dev).unsqueeze(0)  # (1, 3)
            bp = np.asarray(pose["body_pose_params"], dtype=np.float32).copy()
            # Inject locked shape modes (positions 124..129).
            bp[_BODY_POSE_SHAPE_MODE_SLICE] = locked_bp
            bp_t = torch.from_numpy(bp).to(dev).unsqueeze(0)  # (1, >=130)
            hp = pose.get("hand_pose_params")
            hp_t = (
                torch.from_numpy(np.asarray(hp, dtype=np.float32)).to(dev).unsqueeze(0)
                if hp is not None else None
            )

            verts, kpts3d, jc, _, _ = mhr_head._mhr_forward_core(
                global_trans=gt,
                global_rot=gr,
                body_pose_params=bp_t,
                hand_pose_params=hp_t,
                scale_params=t_scale,
                shape_params=t_shape,
                expr_params=t_expr,
                return_keypoints=True,
                slim_mode=True,
            )

            # Repère MHR → repère caméra : flip Y et Z. C'est la
            # "Camera system difference" appliquée dans sam3d_body.py l.3237.
            verts = verts.clone()
            verts[..., [1, 2]] *= -1
            jc = jc.clone()
            jc[..., [1, 2]] *= -1
            if kpts3d is not None:
                kpts3d = kpts3d.clone()
                kpts3d[..., [1, 2]] *= -1
                # Le rig sort 308 keypoints, mais pred_keypoints_3d expose
                # seulement les 70 premiers (cf. sam3d_body.py l.3236).
                kpts3d = kpts3d[:, :70]

            verts_np = verts.squeeze(0).detach().cpu().numpy().astype(np.float32)
            jc_np = jc.squeeze(0).detach().cpu().numpy().astype(np.float32)
            kpts_np = (
                kpts3d.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if kpts3d is not None else None
            )

            results.append({
                "pred_vertices": verts_np,
                "pred_joint_coords": jc_np,
                "pred_keypoints_3d": kpts_np,
            })

    return results
