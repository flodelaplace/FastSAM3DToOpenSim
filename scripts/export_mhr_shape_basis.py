#!/usr/bin/env python3
"""
Export de la BASE DE FORME linéaire du modèle MHR : template (V0/J0/KP0) +
45 directions de forme (dV/dJ/dKP) calculées en POSE DE REPOS, plus un test
empirique de linéarité.

Plan d'appel : identique à `scripts/export_mhr_rest_pose.py` —
  head = estimator.model.head_pose                   # MHRHead body
  head.mhr_forward(
      global_trans=0, global_rot=0,
      body_pose_params=0, hand_pose_params=0,
      scale_params=0, expr_params=0,
      shape_params=<varie>,
      return_keypoints=True, return_joint_coords=True,
  )
  → tuple (verts, j3d_308, jcoords_127)
  Camera flip ([1,2] *= -1), j3d = j3d[:, :70].

Pipeline :
  V0  = mhr_forward(shape=0)                                  # template
  for i in 0..44:
      V_i = mhr_forward(shape=e_i * delta)
      dV[i] = (V_i - V0) / delta                              # direction de shape i
  Test linéarité : pour quelques b ~ N(0,1)^45, comparer
      V_true(b) = mhr_forward(shape=b)
      V_approx  = V0 + Σ b_i * dV[i]
  Rapporte erreur max et RMS en mm.

Sortie .npz :
  V0  [N,3], J0 [127,3], KP0 [70,3], faces [F,3],
  dV  [45,N,3], dJ [45,127,3], dKP [45,70,3],
  delta, meta { coordinate_frame:"mhr_rest", n_shape:45, units:"meters",
                linearity_max_mm, linearity_rms_mm, source:"shape_basis" }

Usage :
    python scripts/export_mhr_shape_basis.py --output outputs/mhr_shape_basis.npz
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


@torch.inference_mode()
def _forward_shape(head, shape_np: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Appelle mhr_forward en POSE DE REPOS avec shape_params = shape_np.
    Retourne (verts, j3d_70, jcoords_127) en numpy float32, repère caméra flippé."""
    B = 1
    shape_params = torch.from_numpy(shape_np.astype(np.float32)).to(device).unsqueeze(0)
    out = head.mhr_forward(
        global_trans=_zeros(B, 3, device),
        global_rot=_zeros(B, 3, device),
        body_pose_params=_zeros(B, 130, device),
        hand_pose_params=_zeros(B, head.num_hand_comps * 2, device),
        scale_params=_zeros(B, head.num_scale_comps, device),
        shape_params=shape_params,
        expr_params=_zeros(B, head.num_face_comps, device),
        return_keypoints=True,
        return_joint_coords=True,
    )
    verts, j3d, jcoords = out
    verts = verts.clone()
    j3d = j3d.clone()
    jcoords = jcoords.clone()
    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1
    jcoords[..., [1, 2]] *= -1
    j3d = j3d[:, :70]
    return (
        verts.squeeze(0).cpu().numpy().astype(np.float32),
        j3d.squeeze(0).cpu().numpy().astype(np.float32),
        jcoords.squeeze(0).cpu().numpy().astype(np.float32),
    )


def main(args):
    print("=== MHR shape basis export + linearity test ===")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {args.device}")
    print(f"  Checkpoint: {args.local_checkpoint}")
    print(f"  Delta:      {args.delta}")
    print(f"  Linearity tests: {args.n_linearity_tests}")
    print()

    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.local_checkpoint,
        detector_name="yolo_pose",
        detector_model=args.detector_model,
        device=args.device,
    )
    head = estimator.model.head_pose
    device = next(head.parameters()).device
    N_shape = head.num_shape_comps  # 45

    # ── Template à shape=0
    print(f"  Computing V0 (shape=0)...")
    V0, KP0, J0 = _forward_shape(head, np.zeros(N_shape), device)
    N_verts = V0.shape[0]
    print(f"    V0={V0.shape}  J0={J0.shape}  KP0={KP0.shape}")

    # ── 45 directions de forme
    dV = np.zeros((N_shape, N_verts, 3), dtype=np.float32)
    dJ = np.zeros((N_shape, 127, 3), dtype=np.float32)
    dKP = np.zeros((N_shape, 70, 3), dtype=np.float32)
    delta = float(args.delta)
    print(f"  Computing {N_shape} shape directions (delta={delta})...")
    for i in range(N_shape):
        e_i = np.zeros(N_shape)
        e_i[i] = delta
        V_i, KP_i, J_i = _forward_shape(head, e_i, device)
        dV[i] = (V_i - V0) / delta
        dJ[i] = (J_i - J0) / delta
        dKP[i] = (KP_i - KP0) / delta
        if (i + 1) % 10 == 0 or i == N_shape - 1:
            print(f"    [{i+1:>2}/{N_shape}] |dV[i]|_max={np.max(np.abs(dV[i]))*1000:.2f} mm")

    # ── Test linéarité
    print()
    print(f"  Linearity test with {args.n_linearity_tests} random betas ~ N(0,1)^{N_shape}...")
    rng = np.random.default_rng(0)
    max_errs = []
    rms_errs = []
    for k in range(args.n_linearity_tests):
        b = rng.standard_normal(N_shape).astype(np.float32)
        V_true, _, _ = _forward_shape(head, b, device)
        V_approx = V0 + np.einsum("s,snc->nc", b, dV)
        err = (V_true - V_approx) * 1000.0  # → mm
        max_mm = float(np.max(np.abs(err)))
        rms_mm = float(np.sqrt(np.mean(err ** 2)))
        max_errs.append(max_mm)
        rms_errs.append(rms_mm)
        print(f"    test {k+1}: max={max_mm:.4f} mm   rms={rms_mm:.4f} mm   ||b||={np.linalg.norm(b):.2f}")

    lin_max_mm = float(max(max_errs))
    lin_rms_mm = float(np.mean(rms_errs))
    print()
    print(f"  → Linearity: WORST-CASE max={lin_max_mm:.4f} mm  | mean RMS={lin_rms_mm:.4f} mm")
    if lin_max_mm < 0.5:
        print(f"    Linéarité quasi-exacte (sub-mm) → base linéaire utilisable telle quelle en numpy")
    elif lin_max_mm < 5.0:
        print(f"    Linéarité acceptable (<5mm) → base linéaire = bonne approximation")
    else:
        print(f"    NON-linéaire significatif (>5mm) → utiliser MHR direct, pas la base seule")

    # ── Faces
    _faces_raw = estimator.faces
    if hasattr(_faces_raw, "detach"):
        _faces_raw = _faces_raw.detach().cpu()
    faces_np = np.asarray(_faces_raw).astype(np.int32)

    # ── Write npz
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        V0=V0, J0=J0, KP0=KP0,
        faces=faces_np,
        dV=dV, dJ=dJ, dKP=dKP,
        delta=np.asarray(delta, dtype=np.float32),
        coordinate_frame=np.asarray("mhr_rest"),
        n_shape=np.asarray(N_shape),
        units=np.asarray("meters"),
        linearity_max_mm=np.asarray(lin_max_mm, dtype=np.float32),
        linearity_rms_mm=np.asarray(lin_rms_mm, dtype=np.float32),
        source=np.asarray("shape_basis"),
    )

    print()
    print(f"OK — shape basis exportée → {args.output}")
    print(f"  V0  : {V0.shape}  dtype={V0.dtype}")
    print(f"  dV  : {dV.shape}  dtype={dV.dtype}")
    print(f"  dJ  : {dJ.shape}")
    print(f"  dKP : {dKP.shape}")
    print(f"  file size : {os.path.getsize(args.output)/1024/1024:.2f} MB")
    print(f"  linearity max_mm = {lin_max_mm:.4f}  rms_mm = {lin_rms_mm:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MHR shape basis + linearity test")
    parser.add_argument("--output", type=str, default="outputs/mhr_shape_basis.npz")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Pas pour la différence finie sur shape_params (défaut 1.0). "
                             "MHR shape est censée être linéaire → tout delta non-nul doit donner la même base.")
    parser.add_argument("--n-linearity-tests", type=int, default=5,
                        help="Nombre de tirages aléatoires b~N(0,1)^45 pour le test de linéarité.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_checkpoint", type=str,
                        default="./checkpoints/sam-3d-body-dinov3")
    parser.add_argument("--detector_model", type=str,
                        default="./checkpoints/yolo/yolo11m-pose.engine")
    args = parser.parse_args()
    main(args)
