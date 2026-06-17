#!/usr/bin/env python3
"""
Export du mesh MHR en POSE NEUTRE (rest pose, paramètres pose=0, forme moyenne)
au même format .npz que l'export existant `--export_mesh_npz` du pipeline vidéo.

Plan d'appel MHR (voir investigation):
  estimator = setup_sam_3d_body(...)              # notebook/utils.py:21
  head      = estimator.model.head_pose           # MHRHead body
  head.mhr_forward(
      global_trans=zeros(B,3),
      global_rot=zeros(B,3),                      # euler XYZ, identity
      body_pose_params=zeros(B,130),
      hand_pose_params=zeros(B, 2*num_hand_comps),
      scale_params=zeros(B,num_scale_comps),      # → scale_mean appliqué
      shape_params=zeros(B,num_shape_comps),      # → forme moyenne
      expr_params=zeros(B,num_face_comps),
      return_keypoints=True, return_joint_coords=True,
  )
  # Apply camera-system flip (lines 933-935 of mhr_head.py)
  verts[...,[1,2]] *= -1 ; j3d[...,[1,2]] *= -1 ; jcoords[...,[1,2]] *= -1
  j3d = j3d[:, :70]                               # 308 → 70 keypoints
  faces = estimator.faces                         # déjà numpy

Sortie .npz identique à `{prefix}_mesh.npz` du pipeline vidéo, sauf que :
  meta.frame_index       = -1
  meta.coordinate_frame  = "mhr_rest"
  meta.pose              = "zero"
  meta.shape             = "mean"  (ou "custom" si --shape passé)
  meta.units             = "meters"
  meta.source            = "rest_pose"

Usage typique :
    python scripts/export_mhr_rest_pose.py --output outputs/mhr_rest_pose.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body


def _zeros(B: int, d: int, device, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(B, d, device=device, dtype=dtype)


def _parse_shape_arg(arg: str | None, num_shape_comps: int) -> np.ndarray:
    """Parse --shape : path vers .npy/.npz OR comma-separated floats OR None."""
    if arg is None:
        return np.zeros(num_shape_comps, dtype=np.float32)
    if os.path.isfile(arg):
        if arg.endswith(".npz"):
            z = np.load(arg)
            v = z.get("shape_params", z.get("shape"))
            if v is None:
                raise ValueError(f"{arg} ne contient pas 'shape_params'/'shape'")
            return np.asarray(v, dtype=np.float32).reshape(-1)
        return np.load(arg).astype(np.float32).reshape(-1)
    return np.asarray([float(x) for x in arg.split(",")], dtype=np.float32)


def main(args):
    print("=== MHR rest pose export ===")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {args.device}")
    print(f"  Checkpoint: {args.local_checkpoint}")
    print()

    # Setup estimator (réutilise le chargement standard du repo)
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.local_checkpoint,
        detector_name="yolo_pose",
        detector_model=args.detector_model,
        device=args.device,
    )

    head = estimator.model.head_pose  # MHRHead body (enable_hand_model=False)
    device = next(head.parameters()).device
    B = 1

    # Pad shape param vector to exactly num_shape_comps
    shape_np = _parse_shape_arg(args.shape, head.num_shape_comps)
    if shape_np.shape[0] > head.num_shape_comps:
        shape_np = shape_np[: head.num_shape_comps]
    elif shape_np.shape[0] < head.num_shape_comps:
        shape_np = np.concatenate(
            [shape_np, np.zeros(head.num_shape_comps - shape_np.shape[0], dtype=np.float32)]
        )
    shape_params = torch.from_numpy(shape_np).to(device).unsqueeze(0)

    print(f"  num_shape_comps : {head.num_shape_comps}")
    print(f"  num_scale_comps : {head.num_scale_comps}")
    print(f"  num_hand_comps  : {head.num_hand_comps}")
    print(f"  num_face_comps  : {head.num_face_comps}")
    print(f"  shape_params    : non-zero={int((shape_np != 0).sum())} / {head.num_shape_comps}")
    print()

    # Tous les params pose à zéro (rest pose)
    global_trans = _zeros(B, 3, device)
    global_rot = _zeros(B, 3, device)          # euler XYZ = identité
    body_pose_params = _zeros(B, 130, device)
    hand_pose_params = _zeros(B, head.num_hand_comps * 2, device)
    scale_params = _zeros(B, head.num_scale_comps, device)
    expr_params = _zeros(B, head.num_face_comps, device)

    with torch.inference_mode():
        out = head.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot,
            body_pose_params=body_pose_params,
            hand_pose_params=hand_pose_params,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
            return_keypoints=True,
            return_joint_coords=True,
        )

    # mhr_forward retourne tuple (verts, keypoints, joint_coords) quand
    # return_keypoints=True ET return_joint_coords=True (cf mhr_head.py:660-674).
    verts, j3d, jcoords = out

    # Camera-system flip (cf mhr_head.py:933-935)
    verts = verts.clone()
    j3d = j3d.clone()
    jcoords = jcoords.clone()
    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1
    jcoords[..., [1, 2]] *= -1

    # j3d sort en [B, 308, 3] (mhr internal keypoints), on garde [:, :70]
    # (cf mhr_head.py:929 : `j3d = j3d[:, :70]`)
    j3d = j3d[:, :70]

    # Squeeze batch dim → numpy
    verts_np = verts.squeeze(0).cpu().numpy().astype(np.float32)
    j3d_np = j3d.squeeze(0).cpu().numpy().astype(np.float32)
    jcoords_np = jcoords.squeeze(0).cpu().numpy().astype(np.float32)

    # Faces : estimator.faces est déjà numpy (cf sam_3d_body_estimator.py:40)
    _faces_raw = estimator.faces
    if hasattr(_faces_raw, "detach"):
        _faces_raw = _faces_raw.detach().cpu()
    faces_np = np.asarray(_faces_raw).astype(np.int32)

    assert verts_np.ndim == 2 and verts_np.shape[1] == 3, f"verts shape {verts_np.shape}"
    assert faces_np.ndim == 2 and faces_np.shape[1] == 3, f"faces shape {faces_np.shape}"
    assert jcoords_np.shape == (127, 3), f"joint_coords shape {jcoords_np.shape}"
    assert j3d_np.shape == (70, 3), f"keypoints shape {j3d_np.shape}"

    # Write .npz, même format que `{prefix}_mesh.npz`
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        verts=verts_np,
        faces=faces_np,
        joint_coords=jcoords_np,
        keypoints=j3d_np,
        frame_index=np.asarray(-1),
        coordinate_frame=np.asarray("mhr_rest"),
        pose=np.asarray("zero"),
        shape=np.asarray("mean" if args.shape is None else "custom"),
        units=np.asarray("meters"),
        n_vertices=np.asarray(int(verts_np.shape[0])),
        source=np.asarray("rest_pose"),
    )

    print(f"OK — MHR rest pose exporté → {args.output}")
    print(f"  verts        : {verts_np.shape}  dtype={verts_np.dtype}")
    print(f"  faces        : {faces_np.shape}  dtype={faces_np.dtype}")
    print(f"  joint_coords : {jcoords_np.shape}")
    print(f"  keypoints    : {j3d_np.shape}")
    print(f"  file size    : {os.path.getsize(args.output)/1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", type=str, default="outputs/mhr_rest_pose.npz",
                        help="Chemin du .npz de sortie.")
    parser.add_argument("--shape", type=str, default=None,
                        help="Vecteur shape_params : chemin .npy/.npz (clé 'shape_params' "
                             "ou 'shape') OU floats comma-separated. Défaut : zéros = forme moyenne.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (défaut: cuda).")
    parser.add_argument("--local_checkpoint", type=str,
                        default="./checkpoints/sam-3d-body-dinov3",
                        help="Local checkpoint dir.")
    parser.add_argument("--detector_model", type=str,
                        default="./checkpoints/yolo/yolo11m-pose.engine",
                        help="Detector model (chargé par setup_sam_3d_body mais inutilisé ici).")
    args = parser.parse_args()
    main(args)
