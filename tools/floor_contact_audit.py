#!/usr/bin/env python3
"""Audit sol + contact sur les TRC deja produits, AVANT / APRES stabilisation.

Mesure les deux criteres de reussite du portage Mesh2Sim :

1. **Dispersion du sol** : hauteur du point de semelle le plus bas a chaque
   POSE (chaque appui), ecart-type et etendue sur l'essai. Sous 2 cm, une
   detection de contact par hauteur devient possible ; au-dessus, non.
2. **Flexion de genou au contact**, avec deux detecteurs :
   - Zeni (celui du pipeline aujourd'hui : `heel_ap - sacrum_ap`), qui depend
     du bassin donc du mode de translation globale ;
   - hauteur plantaire (le portage), qui ne depend d'aucune reference mobile.

L'etat APRES est obtenu en appliquant au TRC la transformation rigide unique
que `ground_plane.stable_floor_transform` calcule (rotation globale + offset
constant). C'est licite : cette transformation est exactement celle que le
drapeau `--stable_floor` applique dans le pipeline, et **les angles articulaires
internes y sont invariants** — le .mot n'a donc pas a etre recalcule. Ce qui
change, c'est QUELLE image est declaree « contact ».

Usage :
    python tools/floor_contact_audit.py outputs/RUN_6224 [outputs/... ...]
    python tools/floor_contact_audit.py --json rapport.json outputs/*/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import direct du module par son chemin : `sam_3d_body/__init__.py` importe
# torch, absent de l'hote (le pipeline tourne en Docker). `ground_plane` est du
# numpy pur et n'a besoin d'aucun de ces paquets.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "_gp", os.path.join(_ROOT, "sam_3d_body", "export", "ground_plane.py"))
_gp = _ilu.module_from_spec(_spec)
sys.modules["_gp"] = _gp  # requis par @dataclass
_spec.loader.exec_module(_gp)
apply_stable_floor = _gp.apply_stable_floor
detect_contact_by_height = _gp.detect_contact_by_height
stable_floor_transform = _gp.stable_floor_transform

try:
    from scipy.signal import butter, filtfilt, find_peaks
except ImportError:  # pragma: no cover
    print("scipy requis", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------

def read_trc(path: str):
    with open(path) as f:
        lines = f.read().split("\n")
    meta = lines[2].split("\t")
    fps = float(meta[0])
    units = meta[4].strip()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    rows = []
    for ln in lines[5:]:
        parts = ln.split("\t")
        if len(parts) < 3 or not parts[0].strip():
            continue
        rows.append([float(x) if x.strip() else np.nan for x in parts])
    A = np.array(rows, dtype=np.float64)
    T = A.shape[0]
    P = A[:, 2:2 + 3 * len(names)].reshape(T, len(names), 3)
    if units == "mm":
        P = P / 1000.0
    return names, P, fps


def read_mot(path: str):
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip() == "endheader")
    cols = lines[i + 1].split("\t")
    rows = []
    for ln in lines[i + 2:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        rows.append([float(x) for x in parts])
    A = np.array(rows, dtype=np.float64)
    return {c.strip(): A[:, k] for k, c in enumerate(cols) if c.strip()}


def _smooth(x, fps, cutoff_hz=8.0):
    nyq = 0.5 * fps
    if cutoff_hz >= nyq or len(x) < 20:
        return np.asarray(x, dtype=np.float64)
    b, a = butter(4, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, np.nan_to_num(x, nan=np.nanmean(x)))


# ---------------------------------------------------------------------------
# metrique 1 : dispersion du sol
# ---------------------------------------------------------------------------

def floor_dispersion(sole_y_by_foot: dict[str, np.ndarray], fps: float,
                     min_cycle_s: float = 0.35):
    """Hauteur du point de semelle le plus bas a chaque POSE.

    Une « pose » = un minimum local de la hauteur du point le plus bas du pied.
    Cette definition ne suppose AUCUN detecteur de contact — c'est justement ce
    qu'on cherche a rendre possible — et elle est identique avant et apres, donc
    comparable.
    """
    vals, per_foot = [], {}
    for foot, y in sole_y_by_foot.items():
        h = _smooth(y, fps)
        idx, _ = find_peaks(-h, distance=max(2, int(min_cycle_s * fps)),
                            prominence=0.02)
        v = y[idx]
        v = v[np.isfinite(v)]
        per_foot[foot] = {"n_poses": int(v.size),
                          "std_cm": float(np.std(v) * 100) if v.size > 1 else float("nan"),
                          "p2p_cm": float(np.ptp(v) * 100) if v.size > 1 else float("nan"),
                          "idx": idx.tolist()}
        vals.append((idx, v))
    allv = np.concatenate([v for _, v in vals]) if vals else np.array([])
    alli = np.concatenate([i for i, _ in vals]) if vals else np.array([])
    out = {
        "n_poses": int(allv.size),
        "std_cm": float(np.std(allv) * 100) if allv.size > 1 else float("nan"),
        "p2p_cm": float(np.ptp(allv) * 100) if allv.size > 1 else float("nan"),
        "mean_cm": float(np.mean(allv) * 100) if allv.size else float("nan"),
        "r_time": (float(np.corrcoef(alli, allv)[0, 1])
                   if allv.size > 2 else float("nan")),
        "per_foot": per_foot,
    }
    return out


# ---------------------------------------------------------------------------
# metrique 2 : flexion de genou au contact
# ---------------------------------------------------------------------------

def zeni_td(heel_ap, toe_ap, sacrum_ap, fps, min_cycle_s=0.4):
    direction = float(np.sign(np.median(toe_ap - heel_ap))) or 1.0
    heel_rel = _smooth(direction * (heel_ap - sacrum_ap), fps)
    prom = 0.30 * (heel_rel.max() - heel_rel.min())
    idx, _ = find_peaks(heel_rel, distance=max(1, int(min_cycle_s * fps)),
                        prominence=prom)
    return idx


def sole_td(sole_y_points: np.ndarray, fps, thresh_m=0.020, min_run_s=0.06):
    """TD = debut du premier run de contact de CHAQUE episode d'appui du pied.

    Chaque point plantaire est mesure contre SON PROPRE plancher (1er centile de
    sa propre serie) — cf. `M2S:contact_optim.py:293-311`. L'episode d'appui du
    pied est l'union des runs de ses points.
    """
    T, P = sole_y_points.shape
    mask = np.zeros(T, dtype=bool)
    for p in range(P):
        y = sole_y_points[:, p]
        if not np.isfinite(y).any():
            continue
        fl = float(np.nanpercentile(y, 1.0))
        for a, b in detect_contact_by_height(y, fl, thresh_m=thresh_m,
                                             min_run_s=min_run_s, fps=fps):
            mask[a:b] = True
    tds, i = [], 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            tds.append(i)
            i = j
        else:
            i += 1
    return np.array(tds, dtype=int), mask


def _stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "values": [round(float(x), 1) for x in v]}


# ---------------------------------------------------------------------------
# pilote
# ---------------------------------------------------------------------------

SOLE_R = [f"SOLE_s{i}_r" for i in range(1, 8)]
SOLE_L = [f"SOLE_s{i}_l" for i in range(1, 8)]


def audit_run(run_dir: str, verbose: bool = True,
              drift_model: str = "linear") -> dict | None:
    trcs = [p for p in glob.glob(os.path.join(run_dir, "*.trc"))
            if "post_ik" not in p]
    if not trcs:
        return None
    names, P, fps = read_trc(trcs[0])
    idx = {n: i for i, n in enumerate(names)}
    if SOLE_R[0] not in idx:
        return None
    ir = [idx[n] for n in SOLE_R if n in idx]
    il = [idx[n] for n in SOLE_L if n in idx]
    plantar = P[:, ir + il, :]

    mots = glob.glob(os.path.join(run_dir, "*_ik.mot"))
    mot = read_mot(mots[0]) if mots else {}

    res = {"run": os.path.basename(run_dir.rstrip("/")),
           "n_frames": int(P.shape[0]), "fps": fps,
           "n_sole": len(ir) + len(il)}

    # marqueurs de controle : on verifie qu'on n'aplatit ni l'oscillation
    # verticale du bassin ni la phase de vol
    ctrl_idx = [idx[n] for n in ("RPSI", "LPSI", "RASI", "LASI") if n in idx]

    def measure(plant, tag, ctrl=None):
        soleY = {"R": np.nanmin(plant[:, :len(ir), 1], axis=1),
                 "L": np.nanmin(plant[:, len(ir):, 1], axis=1)}
        d = floor_dispersion(soleY, fps)
        out = {"floor": d}
        # genou au contact
        knee = {}
        for side, side_pts, cal, toe in (
                ("R", plant[:, :len(ir), 1], "RCAL", "RTOE"),
                ("L", plant[:, len(ir):, 1], "LCAL", "LTOE")):
            k = f"knee_angle_{side.lower()}"
            if k not in mot or cal not in idx:
                continue
            ka = mot[k][:P.shape[0]]
            # axe antero-posterieur = direction de plus grande variance du sacrum
            sac = (P[:, idx["RPSI"], :] + P[:, idx["LPSI"], :]) / 2 \
                if "RPSI" in idx and "LPSI" in idx else P[:, idx["RASI"], :]
            xz = sac[:, [0, 2]] - sac[:, [0, 2]].mean(axis=0)
            _, V = np.linalg.eigh(xz.T @ xz)
            ax = V[:, -1]
            proj = lambda m: m[:, [0, 2]] @ ax  # noqa: E731
            td_z = zeni_td(proj(P[:, idx[cal], :]), proj(P[:, idx[toe], :]),
                           proj(sac), fps)
            td_s, mask = sole_td(side_pts, fps)
            knee[side] = {
                "zeni": {"td_idx": td_z.tolist(),
                         **_stats([ka[i] for i in td_z if i < len(ka)])},
                "sole": {"td_idx": td_s.tolist(),
                         "contact_pct": float(100.0 * mask.mean()),
                         **_stats([ka[i] for i in td_s if i < len(ka)])},
            }
        out["knee_at_td"] = knee
        if ctrl is not None:
            pel = np.nanmean(ctrl[:, :, 1], axis=1)
            low = np.nanmin(plant[:, :, 1], axis=1)
            osc = pel - low                     # hauteur bassin AU-DESSUS du sol local
            flight = float(np.mean(low > 0.05))  # part des images pieds > 5 cm
            out["control"] = {
                "pelvis_above_foot_std_cm": float(np.nanstd(osc)) * 100,
                "pelvis_world_p2p_cm": float(np.nanmax(pel) - np.nanmin(pel)) * 100,
                "flight_fraction": flight,
            }
            if verbose:
                c = out["control"]
                print(f"        controle | oscillation verticale du bassin "
                      f"(monde) {c['pelvis_world_p2p_cm']:.1f} cm | "
                      f"images 2 pieds > 5 cm : {c['flight_fraction']*100:.0f}%")
        if verbose:
            print(f"  [{tag}] sol : {d['n_poses']} poses, "
                  f"std {d['std_cm']:.2f} cm, etendue {d['p2p_cm']:.2f} cm, "
                  f"r(t) {d['r_time']:+.2f}")
            for side, kk in knee.items():
                z, s = kk["zeni"], kk["sole"]
                zs = (f"{z['mean']:.1f}+-{z['std']:.1f} "
                      f"[{z['min']:.0f};{z['max']:.0f}] n={z['n']}"
                      if z.get("n") else "n/a")
                ss = (f"{s['mean']:.1f}+-{s['std']:.1f} "
                      f"[{s['min']:.0f};{s['max']:.0f}] n={s['n']}"
                      if s.get("n") else "n/a")
                print(f"        genou@TD {side} | Zeni {zs} | semelle {ss} "
                      f"(contact {s['contact_pct']:.0f}% des images)")
        return out

    if verbose:
        print(f"== {res['run']}  T={res['n_frames']}  fps={fps:.2f}")
    ctrl = P[:, ctrl_idx, :] if ctrl_idx else None
    res["before"] = measure(plantar, "AVANT", ctrl)

    sf = stable_floor_transform(plantar, fps, per_foot_split=len(ir),
                                min_span_travel_m=2.0, drift_model=drift_model)
    fit, off = sf.fit, sf.offset_m
    res["fit"] = {"ok": fit.ok, "reason": fit.reason, "tilt_deg": fit.tilt_deg,
                  "slope_travel_deg": fit.slope_travel_deg,
                  "slope_cross_deg": fit.slope_cross_deg,
                  "travel_axis_xz": fit.travel_axis_xz,
                  "n_candidates": fit.n_candidates, "n_inliers": fit.n_inliers,
                  "span_travel_m": fit.span_travel_m,
                  "span_cross_m": fit.span_cross_m,
                  "pitch_trusted": fit.pitch_trusted,
                  "roll_trusted": fit.roll_trusted, "offset_m": off,
                  "rotation_applied": sf.rotation_applied,
                  "drift_applied": sf.drift_applied,
                  "drift_cm_per_s": sf.drift_m_per_s * 100,
                  "n_poses": sf.n_poses, "notes": sf.notes}
    if verbose:
        print(f"  [stable] rotation {'OUI' if sf.rotation_applied else 'non'} | "
              f"derive {'OUI' if sf.drift_applied else 'non'} "
              f"[{sf.drift_model}] "
              f"({sf.drift_m_per_s*100:+.2f} cm/s sur {sf.n_poses} poses, "
              f"appui {sf.stance_coverage*100:.0f}%) | "
              f"correction appui max {(0.0 if sf.stance_shift_m is None else float(np.max(np.abs(sf.stance_shift_m))))*100:.1f} cm | "
              f"offset {off*100:+.1f} cm {sf.notes}")
        if fit.ok:
            print(f"  [plan sol] tilt {fit.tilt_deg:.2f}deg | pente progression "
                  f"{fit.slope_travel_deg:+.2f}deg"
                  f"{'' if fit.pitch_trusted else ' (NON OBSERVEE -> 0)'}"
                  f" | pente transverse {fit.slope_cross_deg:+.2f}deg"
                  f"{'' if fit.roll_trusted else ' (NON OBSERVEE -> 0)'} | "
                  f"{fit.n_inliers}/{fit.n_candidates} inliers | etendues "
                  f"{fit.span_travel_m:.2f} m / {fit.span_cross_m:.2f} m | "
                  f"offset {off*100:+.1f} cm")
        else:
            print(f"  [plan sol] ECHEC : {fit.reason}")
    if sf.rotation_applied or sf.drift_applied or abs(off) > 1e-9:
        res["after"] = measure(
            apply_stable_floor(plantar, sf, fps), "APRES",
            apply_stable_floor(ctrl, sf, fps) if ctrl is not None else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--drift", default="linear+stance",
                    choices=["none", "linear", "stance", "linear+stance"],
                    help="modele de derive verticale du sol (defaut: linear+stance, comme --stable_floor)")
    a = ap.parse_args()
    out = []
    for d in a.runs:
        try:
            r = audit_run(d, drift_model=a.drift)
        except Exception as e:  # noqa: BLE001
            print(f"== {d} : ERREUR {type(e).__name__}: {e}")
            continue
        if r:
            out.append(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
