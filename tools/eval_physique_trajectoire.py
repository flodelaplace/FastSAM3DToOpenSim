"""Evaluation hors ligne de sam_3d_body/export/physique_trajectoire.py.

Rejoue l'etage « trajectoire physique » sur les sorties locales et chiffre
avant / apres : derive horizontale et de profondeur en vol, respect de la
parabole, glissement des pieds en appui, deplacement net, vitesses et
accelerations, et rigidite (aucun angle ne doit changer).

Deux sources :
  * ``_etapes_marqueurs.npz`` (SYNKRO_DUMP_ETAPES=1) : marqueurs AVANT lissage
    et anti-glissement. On rejoue les etages EXISTANTS avec les reglages de
    demo_video_opensim.py pour le module, puis l'etage physique par-dessus ;
  * TRC pre-IK ``markers_<nom>.trc`` : deja passes par les etages existants.

Les phases (appuis / vols) sont detectees UNE fois sur l'entree de l'etage
physique et reutilisees pour les metriques avant et apres : on juge la
trajectoire, pas la detection.

Usage :
    python3 tools/eval_physique_trajectoire.py [--json sortie.json] [--dj DIR ...]

Hote sans torch : les modules sont charges par chemin, sans importer le
paquet sam_3d_body.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _charger(nom, rel):
    spec = importlib.util.spec_from_file_location(nom, os.path.join(RACINE, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nom] = mod
    spec.loader.exec_module(mod)
    return mod


pt = _charger("physique_trajectoire", "sam_3d_body/export/physique_trajectoire.py")
ct = _charger("coordinate_transform_eval", "sam_3d_body/export/coordinate_transform.py")


def lire_trc(path):
    with open(path) as f:
        lignes = f.read().splitlines()
    v = lignes[2].split("\t")
    fps, unite = float(v[0]), v[4]
    noms = [n for n in lignes[3].split("\t")[2:] if n.strip()]
    rows = []
    for l in lignes[5:]:
        if not l.strip():
            continue
        p = l.split("\t")
        r = [float(x) if x.strip() not in ("", "nan", "NaN") else np.nan
             for x in p[2:2 + 3 * len(noms)]]
        r += [np.nan] * (3 * len(noms) - len(r))
        rows.append(r)
    M = np.array(rows).reshape(len(rows), len(noms), 3)
    if unite.lower().startswith("mm"):
        M = M / 1000.0
    return M, noms, fps


# (essai, source, module du pipeline, camera portee, geste)
ESSAIS_DUMP = [
    ("DIAG_basket", None, False, "basket libre"),
    ("DIAG_sprint", "d3.sprint_start", False, "depart sprint"),
    ("DIAG_run", "d3.running", True, "course camera portee"),
    ("CAP_run_hh", "d3.running", True, "course camera portee"),
    ("FIX_run_hh", "d3.running", True, "course camera portee"),
    ("DIAG_squat", "d3.squat", False, "squat (controle)"),
    ("DIAG_appsquat", "d3.squat", False, "squat (controle)"),
]
ESSAIS_TRC = [
    ("NUIT_jump", "CMJ"),
    ("MO_cmj", "CMJ"),
    ("FINAL_basket", "basket libre"),
    ("LIBRE_basket", "basket libre"),
    ("SOL2_basket", "basket libre"),
    ("NUIT_sprint", "depart sprint"),
    ("OSC_run_hh", "course camera portee"),
    ("FIX_gait", "marche (controle)"),
    ("NUIT_gait", "marche (controle)"),
    ("NUIT_hop", "hop unipodal"),
]


def etages_existants(M, noms, fps, module, portee):
    """Rejoue lissage rigide + anti-glissement comme demo_video_opensim.py."""
    out = M.copy()
    traj = module is None or module in ("d3.gait", "d3.running", "d3.sprint_start",
                                        "d3.single_leg_hop")
    if traj:
        r, _ = ct.lisser_trajectoire_rigide(out, noms, fps=fps)
        out = r if r is not None else out
    if not portee:
        en_place = module in ("d3.squat", "d3.sit_to_stand", "d3.single_leg_squat",
                              "d3.jump", "d3.drop_jump")
        out, _ = ct.anti_foot_skate_markers(out, noms, fps=fps, en_place=en_place,
                                            shift_lowpass_hz=8.0 if module is None else 2.0)
    return out


def _angle(a, b, c):
    u, v = a - b, c - b
    cs = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def rigidite(M0, M1, noms):
    """Ecart max a une translation commune (mm) et variation d'angles (deg)."""
    d = M1 - M0
    med = np.nanmedian(d, axis=1, keepdims=True)
    ecart = float(np.nanmax(np.abs(d - med)) * 1000) if np.isfinite(d).any() else 0.0
    ix = {n: i for i, n in enumerate(noms)}
    dang = 0.0
    for S in ("R", "L"):
        try:
            h = np.nanmean(M0[:, [ix[S + "ASI"], ix[S + "PSI"]]], 1)
            k = np.nanmean(M0[:, [ix[S + "LFC"], ix[S + "MFC"]]], 1)
            a = np.nanmean(M0[:, [ix[S + "LMAL"], ix[S + "MMAL"]]], 1)
            h1 = np.nanmean(M1[:, [ix[S + "ASI"], ix[S + "PSI"]]], 1)
            k1 = np.nanmean(M1[:, [ix[S + "LFC"], ix[S + "MFC"]]], 1)
            a1 = np.nanmean(M1[:, [ix[S + "LMAL"], ix[S + "MMAL"]]], 1)
        except KeyError:
            continue
        dang = max(dang, float(np.nanmax(np.abs(_angle(h, k, a) - _angle(h1, k1, a1)))))
    return ecart, dang


def resumer(m):
    v = m["vols"]
    r = {
        "n_vols": len(v),
        "ecart_x_cm": float(np.median([x["ecart_x_cm"] for x in v])) if v else None,
        "ecart_x_max_cm": float(np.max([x["ecart_x_cm"] for x in v])) if v else None,
        "ecart_z_cm": float(np.median([x["ecart_z_cm"] for x in v])) if v else None,
        "rms_parab_cm": float(np.median([x["rms_parabole_cm"] for x in v])) if v else None,
        "saut_v_ms": float(np.nanmedian([x["saut_vitesse_ms"] for x in v])) if v else None,
        "glisse_med_cm": m["glissement_cm"]["mediane"],
        "glisse_p90_cm": m["glissement_cm"]["p90"],
        "glisse_v_ms": m["glissement_cm"]["vitesse_mediane_ms"],
        "depl_net_m": m.get("deplacement_net_m"),
        "v_p99": m.get("vitesse_h", {}).get("p99"),
        "a_p99": m.get("acceleration_h", {}).get("p99"),
        "a_max": m.get("acceleration_h", {}).get("max"),
    }
    return r


def evaluer(nom, geste, M_in, noms, fps, portee, variantes, M_brut=None):
    ph = pt.detecter_phases(M_in, noms, fps, sol="glissant" if portee else "global")
    res = {"essai": nom, "geste": geste, "fps": fps, "T": int(M_in.shape[0]),
           "vols_detectes": [(round(v["debut"] / fps, 2), round(v["duree_s"], 3), v["bord"])
                             for v in ph["vols"]],
           "rejets": ph["rejets"], "part_appui": float(ph["appui"].mean())}
    if M_brut is not None:
        res["brut"] = resumer(pt.metriques_trajectoire(M_brut, noms, fps, ph))
    res["existant"] = resumer(pt.metriques_trajectoire(M_in, noms, fps, ph))
    for vn, kw in variantes.items():
        out, s, rap = pt.trajectoire_physique(M_in, noms, fps, phases=ph, **kw)
        r = resumer(pt.metriques_trajectoire(out, noms, fps, ph))
        r["applique"] = rap["applique"]
        r["corr_max_cm"] = rap.get("correction_max_cm")
        r["corr_h_cm"] = rap.get("correction_max_horizontale_cm")
        r["corr_v_cm"] = rap.get("correction_max_verticale_cm")
        r["bornes"] = rap.get("bornes")
        r["avert"] = rap["avertissements"]
        r["rigid_mm"], r["dangle_deg"] = rigidite(M_in, out, noms)
        res[vn] = r
    return res


VARIANTES = {
    "physique": {},
    "physique_explosif": {"a_max_ms2": 25.0},
    "physique_sans_bornes": {"borner": False},
    "physique_ancre": {"ancrer_appuis": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--dj", nargs="*", default=[], help="dossiers de drop jumps (TRC pre-IK)")
    a = ap.parse_args()
    out = []
    for nom, module, portee, geste in ESSAIS_DUMP:
        f = os.path.join(RACINE, "outputs", nom, "_etapes_marqueurs.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        M, noms, fps = z["markers"].astype(float), [str(x) for x in z["names"]], float(z["fps"])
        M_e = etages_existants(M, noms, fps, module, portee)
        out.append(evaluer(nom + " (dump)", geste, M_e, noms, fps, portee, VARIANTES, M_brut=M))
        print("ok", nom, file=sys.stderr)
    for nom, geste in ESSAIS_TRC:
        cand = [p for p in glob.glob(os.path.join(RACINE, "outputs", nom, "markers_*.trc"))
                if not p.endswith("_post_ik.trc")]
        if not cand:
            continue
        M, noms, fps = lire_trc(cand[0])
        out.append(evaluer(nom, geste, M, noms, fps, "run_hh" in nom, VARIANTES))
        print("ok", nom, file=sys.stderr)
    for d in a.dj:
        cand = [p for p in glob.glob(os.path.join(d, "markers_*.trc"))
                if not p.endswith("_post_ik.trc")]
        if not cand:
            continue
        M, noms, fps = lire_trc(cand[0])
        out.append(evaluer(os.path.basename(d.rstrip("/")), "drop jump", M, noms, fps, False,
                           VARIANTES))
        print("ok", d, file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
    for r in out:
        print(f"\n=== {r['essai']} — {r['geste']} ({r['T']} images, {r['fps']:.0f} i/s, "
              f"appui {r['part_appui'] * 100:.0f} %)")
        print("  vols :", r["vols_detectes"], "| rejets :", r["rejets"])
        for k in ("brut", "existant", *VARIANTES):
            if k not in r:
                continue
            x = r[k]
            f = lambda v, n=1: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"
            extra = ""
            if "corr_max_cm" in x:
                extra = (f" | corr {f(x['corr_h_cm'])}/{f(x['corr_v_cm'])} cm h/v"
                         f" | bornes {x['bornes']} | rigide {x['rigid_mm']:.3f} mm "
                         f"{x['dangle_deg']:.2e} deg {x['avert'] or ''}")
            print(f"  {k:22s} vol: dX {f(x['ecart_x_cm'])} (max {f(x['ecart_x_max_cm'])}) "
                  f"dZ {f(x['ecart_z_cm'])} parab {f(x['rms_parab_cm'])} dv {f(x['saut_v_ms'], 2)}"
                  f" | glisse {f(x['glisse_med_cm'])}/{f(x['glisse_p90_cm'])} v{f(x['glisse_v_ms'], 2)} | net "
                  f"{f(x['depl_net_m'], 3)} m | v99 {f(x['v_p99'], 2)} a99 {f(x['a_p99'])} "
                  f"amax {f(x['a_max'])}{extra}")


if __name__ == "__main__":
    main()
