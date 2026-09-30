"""Tests de sam_3d_body/export/physique_trajectoire.py sur signaux synthetiques.

Le module est charge par son chemin : importer le paquet ``sam_3d_body``
tirerait torch, absent de l'hote.

    python3 -m pytest tests/test_physique_trajectoire.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "sam_3d_body", "export", "physique_trajectoire.py")
_spec = importlib.util.spec_from_file_location("physique_trajectoire", _P)
pt = importlib.util.module_from_spec(_spec)
sys.modules["physique_trajectoire"] = pt
_spec.loader.exec_module(pt)

FPS = 60.0
G = pt.G

# Corps « en T » simplifie : decalages (m) par rapport au milieu du bassin, en
# station debout (bassin a 1,0 m). X avant (= profondeur camera), Y haut,
# Z lateral droit.
_CORPS = {
    "c_head": (0.0, 0.65, 0.0),
    "RACR": (0.0, 0.45, 0.18), "LACR": (0.0, 0.45, -0.18),
    "RASI": (0.06, 0.0, 0.12), "LASI": (0.06, 0.0, -0.12),
    "RPSI": (-0.08, 0.02, 0.05), "LPSI": (-0.08, 0.02, -0.05),
    "RLEL": (0.0, 0.15, 0.22), "RMEL": (0.0, 0.15, 0.16),
    "LLEL": (0.0, 0.15, -0.22), "LMEL": (0.0, 0.15, -0.16),
    "RFAradius": (0.0, -0.1, 0.22), "RFAulna": (0.0, -0.1, 0.18),
    "LFAradius": (0.0, -0.1, -0.22), "LFAulna": (0.0, -0.1, -0.18),
    "RIndex": (0.0, -0.2, 0.22), "RPinky": (0.0, -0.2, 0.18),
    "LIndex": (0.0, -0.2, -0.22), "LPinky": (0.0, -0.2, -0.18),
    "RLFC": (0.0, -0.5, 0.14), "RMFC": (0.0, -0.5, 0.06),
    "LLFC": (0.0, -0.5, -0.14), "LMFC": (0.0, -0.5, -0.06),
}
# Pied, par rapport a un point au sol sous la cheville (m).
_PIED = {
    "LMAL": (0.0, 0.07, 0.04), "MMAL": (0.0, 0.07, -0.03),
    "CAL": (-0.05, 0.03, 0.0), "TOE": (0.18, 0.03, 0.0), "MT5": (0.13, 0.03, 0.04),
    "SOLE_s1": (-0.04, 0.0, 0.0), "SOLE_s2": (0.08, 0.0, 0.03), "SOLE_s3": (0.17, 0.0, 0.0),
}


def _noms():
    noms = list(_CORPS)
    for S, suf in (("R", "_r"), ("L", "_l")):
        for k in _PIED:
            noms.append(k + suf if k.startswith("SOLE") else S + k)
    return noms


def _construire(bassin, pied_d, pied_g):
    """Marqueurs (T, N, 3) : corps rigide sur ``bassin`` (T,3), pieds poses sur
    ``pied_d`` / ``pied_g`` (T,3, point au sol sous la cheville)."""
    noms = _noms()
    T = bassin.shape[0]
    M = np.zeros((T, len(noms), 3))
    for i, n in enumerate(noms):
        if n in _CORPS:
            M[:, i] = bassin + np.asarray(_CORPS[n])
        else:
            cote = "d" if (n.endswith("_r") or n.startswith("R")) else "g"
            base = n[:-2] if n.startswith("SOLE") else n[1:]
            off = np.asarray(_PIED[base]) * (1, 1, 1 if cote == "d" else -1)
            ref = pied_d if cote == "d" else pied_g
            M[:, i] = ref + off
    return M, noms


def _saut(t_appui=0.5, t_vol=0.5, derive_x=0.0, fps=FPS):
    """CMJ simplifie : debout, vol balistique, debout. Corps entierement rigide
    (les pieds montent avec le bassin). ``derive_x`` : bosse de profondeur
    injectee pendant le vol (erreur de cam_t), en metres."""
    n_a = int(round(t_appui * fps))
    n_v = int(round(t_vol * fps))
    T = 2 * n_a + n_v
    y = np.full(T, 1.0)
    tv = np.arange(1, n_v + 1) / fps
    D = n_v / fps + 1.0 / fps          # du dernier appui a la reception
    v0 = G * D / 2.0
    y[n_a:n_a + n_v] = 1.0 + v0 * tv - 0.5 * G * tv ** 2
    x = np.zeros(T)
    bosse = np.zeros(T)
    bosse[n_a:n_a + n_v] = derive_x * np.sin(np.pi * tv / D)
    bassin_vrai = np.column_stack([x, y, np.zeros(T)])
    bassin = bassin_vrai + np.column_stack([bosse, np.zeros(T), np.zeros(T)])
    decal = bassin - np.array([0.0, 1.0, 0.0])
    pied_d = np.column_stack([np.zeros(T), np.zeros(T), np.full(T, 0.12)]) + decal
    pied_g = np.column_stack([np.zeros(T), np.zeros(T), np.full(T, -0.12)]) + decal
    M, noms = _construire(bassin, pied_d, pied_g)
    return M, noms, (n_a, n_a + n_v), bassin_vrai


def _marche(duree=4.0, v=1.2, fps=FPS, bruit=0.0, seed=0):
    """Marche : pieds alternes (appui 60 %, double appui), bassin a vitesse
    constante, pied en oscillation leve de 8 cm."""
    T = int(round(duree * fps))
    t = np.arange(T) / fps
    cycle = 1.1
    rng = np.random.default_rng(seed)
    x_b = v * t + bruit * rng.standard_normal(T)
    bassin = np.column_stack([x_b, np.full(T, 1.0), np.zeros(T)])
    L = v * cycle                                  # longueur de foulee

    def pied(phase0, z):
        ph = ((t / cycle) + phase0) % 1.0
        n_cyc = np.floor((t / cycle) + phase0)
        x = np.empty(T)
        y = np.zeros(T)
        appui = ph < 0.6
        # en appui : fixe ; en oscillation : avance de L en 40 % du cycle
        x[appui] = n_cyc[appui] * L
        u = (ph[~appui] - 0.6) / 0.4
        x[~appui] = n_cyc[~appui] * L + L * (0.5 - 0.5 * np.cos(np.pi * u))
        y[~appui] = 0.08 * np.sin(np.pi * u)
        return np.column_stack([x - 0.3 * L, y, np.full(T, z)])

    M, noms = _construire(bassin, pied(0.0, 0.12), pied(0.5, -0.12))
    return M, noms


def _com(M, noms):
    return pt.centre_de_masse_approx(M, noms)


# ---------------------------------------------------------------------------


def test_saut_detecte_un_vol():
    M, noms, (a, b), _ = _saut()
    ph = pt.detecter_phases(M, noms, FPS)
    assert len(ph["vols"]) == 1
    v = ph["vols"][0]
    # detection a +-2 images (seuil de 3 cm sur la semelle)
    assert abs(v["debut"] - a) <= 3 and abs(v["fin"] - b) <= 3
    assert not v["bord"]


def test_saut_derive_de_profondeur_corrigee():
    M, noms, (a, b), vrai = _saut(derive_x=0.10)
    out, s, rap = pt.trajectoire_physique(M, noms, FPS)
    assert rap["applique"] and s is not None
    c_in, c_out = _com(M, noms), _com(out, noms)
    ph = pt.detecter_phases(M, noms, FPS)
    v = ph["vols"][0]
    seg = c_out[v["debut"] - 1:v["fin"] + 1, 0]
    droite = np.linspace(seg[0], seg[-1], seg.size)
    # en vol, la profondeur suit une droite (vitesse constante)...
    assert np.abs(seg - droite).max() < 0.005
    # ... alors qu'elle s'en ecartait de pres de 10 cm
    seg0 = c_in[v["debut"] - 1:v["fin"] + 1, 0]
    assert np.abs(seg0 - np.linspace(seg0[0], seg0[-1], seg0.size)).max() > 0.08
    # et la verite (profondeur constante) est retrouvee a 1,5 cm pres
    com_vrai = c_in[:, 0] - (M[:, 0, 0] - (vrai[:, 0] + _CORPS["c_head"][0]))
    assert np.abs(c_out[:, 0] - com_vrai).max() < 0.015


def test_saut_verticale_parabolique_et_appuis_intacts():
    M, noms, (a, b), _ = _saut(derive_x=0.10)
    # on abime la verticale du vol : aplatie de 20 %
    M2 = M.copy()
    ph = pt.detecter_phases(M2, noms, FPS)
    out, s, rap = pt.trajectoire_physique(M2, noms, FPS, phases=ph)
    cy = _com(out, noms)[:, 1]
    v = ph["vols"][0]
    seg = cy[v["debut"] - 1:v["fin"] + 1]
    tt = np.arange(seg.size) / FPS
    c2 = np.polyfit(tt, seg, 2)[0]
    assert abs(-2 * c2 - G) < 0.2                     # parabole a g
    # loin du vol, le corps ne bouge pas (appui debout)
    assert np.abs(s[:max(1, a - 12)]).max() < 0.005
    assert np.abs(s[b + 20:]).max() < 0.01
    # verticale nulle en appui : le pied pose reste au sol
    assert np.abs(s[ph["appui"], 1]).max() < 0.005


def test_translation_rigide_aucun_angle_ne_change():
    M, noms, _, _ = _saut(derive_x=0.10)
    out, s, _ = pt.trajectoire_physique(M, noms, FPS)
    d = out - M
    # meme vecteur pour tous les marqueurs d'une image
    assert np.abs(d - d[:, :1, :]).max() < 1e-9
    assert np.allclose(d[:, 0, :], s, atol=1e-9)
    # distances inter-marqueurs inchangees
    i, j = noms.index("RASI"), noms.index("RTOE")
    assert np.allclose(np.linalg.norm(M[:, i] - M[:, j], axis=1),
                       np.linalg.norm(out[:, i] - out[:, j], axis=1), atol=1e-9)


def test_marche_inchangee_a_quelques_mm():
    M, noms = _marche()
    ph = pt.detecter_phases(M, noms, FPS)
    assert ph["vols"] == []                           # toujours un pied au sol
    out, s, rap = pt.trajectoire_physique(M, noms, FPS, phases=ph)
    assert rap["applique"]
    assert np.abs(s).max() < 0.003
    # deplacement net conserve
    c0, c1 = _com(M, noms), _com(out, noms)
    assert abs((c1[-1, 0] - c1[0, 0]) - (c0[-1, 0] - c0[0, 0])) < 0.003


def test_marche_bruitee_deplacement_conserve():
    M, noms = _marche(bruit=0.01, seed=3)
    out, s, rap = pt.trajectoire_physique(M, noms, FPS)
    m0 = pt.metriques_trajectoire(M, noms, FPS, pt.detecter_phases(M, noms, FPS))
    m1 = pt.metriques_trajectoire(out, noms, FPS, pt.detecter_phases(M, noms, FPS))
    assert abs(m1["deplacement_net_m"] - m0["deplacement_net_m"]) < 0.02
    # les bornes d'acceleration ont retire la gigue
    assert m1["acceleration_h"]["p99"] < m0["acceleration_h"]["p99"]


def test_bord_de_clip_ignore_par_defaut():
    M, noms, (a, b), _ = _saut(t_appui=0.5, derive_x=0.08)
    M = M[a + 5:]                                      # le clip commence en l'air
    ph = pt.detecter_phases(M, noms, FPS)
    assert all(v["bord"] for v in ph["vols"])
    out, s, rap = pt.trajectoire_physique(M, noms, FPS, phases=ph)
    assert rap["applique"] and rap["vols_corriges"] == []
    assert np.all(np.isfinite(s))
    assert np.abs(s).max() < 0.02
    # sur demande, la contrainte horizontale s'applique sans planter
    out2, s2, rap2 = pt.trajectoire_physique(M, noms, FPS, phases=ph, vols_en_bord=True)
    assert rap2["applique"] and np.all(np.isfinite(s2))


def test_donnees_manquantes():
    M, noms, (a, b), _ = _saut(derive_x=0.10)
    M = M.copy()
    M[3:6] = np.nan                                    # images entierement perdues
    M[a + 5, noms.index("c_head")] = np.nan            # un marqueur perdu en vol
    M[a + 8, :10] = np.nan                             # moitie du corps perdue
    out, s, rap = pt.trajectoire_physique(M, noms, FPS)
    assert rap["applique"]
    assert np.all(np.isfinite(s))
    assert np.all(np.isnan(out[3:6]))                  # les trous restent des trous
    fin = np.isfinite(M)
    assert np.all(np.isfinite(out[fin]))
    ph = pt.detecter_phases(M, noms, FPS)
    assert len(ph["vols"]) == 1


def test_unites_millimetres():
    M, noms, _, _ = _saut(derive_x=0.10)
    out_m, s_m, _ = pt.trajectoire_physique(M, noms, FPS)
    out_mm, s_mm, _ = pt.trajectoire_physique(M * 1000.0, noms, FPS)
    assert np.allclose(s_m, s_mm, atol=1e-6)           # decalages toujours en metres
    assert np.allclose(out_mm / 1000.0, out_m, atol=1e-6)


def test_derive_verticale_lente_n_est_pas_un_vol():
    """Toute la scene monte de 12 cm en 0,6 s (defaut de sol), sans vol."""
    M, noms = _marche(duree=3.0, v=0.0)
    M = M.copy()
    T = M.shape[0]
    rampe = np.zeros(T)
    i0, i1 = int(1.0 * FPS), int(1.6 * FPS)
    rampe[i0:i1] = np.linspace(0, 0.12, i1 - i0)
    rampe[i1:int(1.8 * FPS)] = 0.12
    rampe[int(1.8 * FPS):int(2.2 * FPS)] = np.linspace(0.12, 0, int(2.2 * FPS) - int(1.8 * FPS))
    M[:, :, 1] += rampe[:, None]
    ph = pt.detecter_phases(M, noms, FPS)
    for v in ph["vols"]:
        assert v["duree_s"] < 0.15, v                  # rien de long n'est accepte


def test_saut_de_suivi_en_vol_laisse_tel_quel():
    M, noms, (a, b), _ = _saut(derive_x=0.0)
    M = M.copy()
    M[a + 10:a + 14, :, 0] += 0.40                     # 40 cm de saut en plein vol
    out, s, rap = pt.trajectoire_physique(M, noms, FPS, borner=False)
    assert rap["vols_corriges"] == []
    assert any("saut de suivi" in w for w in rap["avertissements"])


def test_borne_acceleration_retire_un_pic_sans_deriver():
    M, noms = _marche(duree=4.0, v=1.0)
    M = M.copy()
    k = int(2.0 * FPS)
    M[k:k + 4, :, 0] += 0.25                           # pic de suivi de 25 cm
    ph = pt.detecter_phases(M, noms, FPS)
    out, s, rap = pt.trajectoire_physique(M, noms, FPS, phases=ph)
    assert rap["bornes"]["acceleration_images"] > 0
    m0 = pt.metriques_trajectoire(M, noms, FPS, ph)
    m1 = pt.metriques_trajectoire(out, noms, FPS, ph)
    assert m1["acceleration_h"]["max"] < 0.5 * m0["acceleration_h"]["max"]
    assert abs(m1["deplacement_net_m"] - m0["deplacement_net_m"]) < 0.01
    assert np.abs(s[:int(1.0 * FPS)]).max() < 0.01     # loin du pic : intact


def test_appliquer_decalages_liste_et_reechantillonnage():
    d = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    pts = [np.zeros((2, 3)), None, np.ones((2, 3))]
    out = pt.appliquer_decalages(pts, d)
    assert out[1] is None
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[2], 1.0 + np.array([0.1, 0.2, 0.3]))
    assert np.allclose(pts[2], 1.0)                    # l'entree n'est pas modifiee
    arr = pt.appliquer_decalages(np.zeros((2, 4, 3)), d)
    assert np.allclose(arr[1], [0.1, 0.2, 0.3])
    assert pt.appliquer_decalages(pts, None) is pts


def test_entrees_degenerees():
    out, s, rap = pt.trajectoire_physique(np.zeros((3, 2, 3)), ["a", "b"], FPS)
    assert s is None and not rap["applique"]
    M, noms, _, _ = _saut()
    ph = pt.detecter_phases(M[:, :5], noms[:5], FPS)   # pas de pied
    assert ph["vols"] == [] and ph["rejets"]


def test_sequence_longue_rapide():
    import time
    M, noms = _marche(duree=30.0, bruit=0.005)          # 1800 images
    t0 = time.time()
    out, s, rap = pt.trajectoire_physique(M, noms, FPS)
    assert rap["applique"]
    assert time.time() - t0 < 30.0
