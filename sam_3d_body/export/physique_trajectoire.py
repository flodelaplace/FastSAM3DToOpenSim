"""Trajectoire globale PHYSIQUEMENT COHERENTE, par translation rigide. PROTOTYPE.

Ce module n'est branche NULLE PART. Plan d'integration et chiffres :
``docs/physique_trajectoire_2026-09-30.md``.

Demande de Florian (2026-09-30) : « tres important de gerer les contacts au
sol, l'anti-glissement, et quand il y a un mouvement et un saut, avoir un
controle de la profondeur plausible pour eviter que la personne recule alors
qu'elle saute ; integrer un peu de physique du corps et de coherence
temporelle ».

LE CONSTAT. La translation globale du sujet vient de ``cam_t``, c'est-a-dire
d'une PROFONDEUR estimee image par image. Sur ``outputs/NUIT_jump`` (CMJ sur
place, 30 i/s) le bassin recule de 10 cm en profondeur pendant la montee du
vol, puis avance de 16 cm a la descente : aucune force horizontale n'agit sur
un corps en l'air, ce mouvement est impossible. C'est l'erreur de profondeur
qui se lit comme un deplacement.

CE QUE FAIT CET ETAGE. Il cherche un decalage RIGIDE s(t) (un vecteur 3D par
image, le meme pour tous les marqueurs) qui rend la trajectoire du centre de
masse compatible avec la mecanique du point :

  1. EN VOL (aucune semelle au sol) aucune force horizontale : l'acceleration
     horizontale du centre de masse est NULLE (vitesse constante en X et Z,
     profondeur comprise), et l'acceleration verticale vaut -g.
  2. EN APPUI le pied pose ne glisse pas (optionnel, voir plus bas).
  3. PARTOUT la vitesse et l'acceleration horizontales restent humaines, et le
     raccord appui/vol est C1 (la vitesse est continue au decollage : elle ne
     change que sous l'effet d'une force, donc en appui).
  4. Et, faute de mieux, la correction reste PETITE : le deplacement mesure par
     ``cam_t`` reste la reference hors des contraintes ci-dessus.

Tout est QUADRATIQUE en s, separable par axe (les bornes, non lineaires, sont
traitees par projections successives) : moindres carres creux, sans gradient
ni solveur iteratif lourd.

Deux proprietes decoulent du choix d'une translation rigide, et sont la raison
de ce choix (meme argument que ``lisser_trajectoire_rigide``) :
  * AUCUN ANGLE ARTICULAIRE NE PEUT CHANGER : une translation commune ne
    modifie aucune distance ni aucun angle entre marqueurs ;
  * les decalages se rejouent tels quels sur le mesh GLB, comme ceux de
    l'anti-glissement.

⚠️ CENTRE DE MASSE ET NON BASSIN. En vol, c'est le centre de masse qui suit
une droite (et une parabole), pas le bassin : quand le sujet groupe les jambes
ou lance les bras, le bassin bouge par rapport au centre de masse. On
l'approche par une moyenne des segments ponderee par les fractions de masse de
de Leva (1996) — une translation rigide ne change pas la position relative du
centre de masse, donc l'erreur d'approximation ne se reporte que sur la forme
imposee, pas sur la posture.

⚠️ ANCRAGE DES APPUIS. La formulation « variance d'appui » (le pied est
immobile pendant chaque appui, sa position est libre) est disponible
(``ancrer_appuis=True``), mais elle a deja ete mesuree dans le pipeline
(`outputs/GAIT_VAR`, 2026-09-09) : quand les appuis se chevauchent (marche),
elle reconstruit la trajectoire depuis les pieds et gonfle le deplacement de
26 %. Elle est donc COUPEE PAR DEFAUT : l'etage est concu pour passer APRES
``anti_foot_skate_markers``, qui gere deja l'appui avec des reglages par
geste, et pour ne toucher qu'au vol, aux raccords et aux bornes.

Convention : marqueurs (T, N, 3) Y vers le haut, en metres ou en millimetres
(detection automatique comme dans coordinate_transform.py) ; les decalages
rendus sont toujours en METRES.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "G",
    "centre_de_masse_approx",
    "detecter_phases",
    "trajectoire_physique",
    "appliquer_decalages",
    "metriques_trajectoire",
]

G = 9.81

# Fractions de masse segmentaires, de Leva (1996), homme — table 4. La femme
# differe de moins de 2 points par segment, sans effet sur une translation.
# (segment, fraction, marqueurs proximaux, marqueurs distaux, position du
# centre de masse le long du segment depuis le proximal).
_SEGMENTS = [
    ("tete", 0.0694, ("c_head",), None, 0.0),
    ("tronc", 0.4346, ("RACR", "LACR"), ("RASI", "LASI", "RPSI", "LPSI"), 0.51),
    ("bras_d", 0.0271, ("RACR",), ("RLEL", "RMEL"), 0.577),
    ("bras_g", 0.0271, ("LACR",), ("LLEL", "LMEL"), 0.577),
    ("avbras_d", 0.0162, ("RLEL", "RMEL"), ("RFAradius", "RFAulna"), 0.457),
    ("avbras_g", 0.0162, ("LLEL", "LMEL"), ("LFAradius", "LFAulna"), 0.457),
    ("main_d", 0.0061, ("RFAradius", "RFAulna"), ("RIndex", "RPinky"), 0.79),
    ("main_g", 0.0061, ("LFAradius", "LFAulna"), ("LIndex", "LPinky"), 0.79),
    ("cuisse_d", 0.1416, ("RASI", "RPSI"), ("RLFC", "RMFC"), 0.41),
    ("cuisse_g", 0.1416, ("LASI", "LPSI"), ("LLFC", "LMFC"), 0.41),
    ("jambe_d", 0.0433, ("RLFC", "RMFC"), ("RLMAL", "RMMAL"), 0.44),
    ("jambe_g", 0.0433, ("LLFC", "LMFC"), ("LLMAL", "LMMAL"), 0.44),
    ("pied_d", 0.0137, ("RCAL", "RTOE", "RMT5"), None, 0.0),
    ("pied_g", 0.0137, ("LCAL", "LTOE", "LMT5"), None, 0.0),
]
_BASSIN = ("RASI", "LASI", "RPSI", "LPSI")


def _unite(markers: np.ndarray) -> float:
    """1000 si les marqueurs sont en millimetres, 1 s'ils sont en metres."""
    with np.errstate(all="ignore"):
        m = np.nanmax(np.abs(markers)) if np.isfinite(markers).any() else 0.0
    return 1000.0 if m > 50.0 else 1.0


def _moyenne(markers, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(markers[:, idx, :], axis=1)


def centre_de_masse_approx(markers: np.ndarray, marker_names) -> np.ndarray:
    """Centre de masse approche (T, 3), dans l'unite des marqueurs.

    Moyenne des centres de segments ponderee par les fractions de masse de de
    Leva. Un segment dont un marqueur manque est retire et les poids restants
    renormalises, image par image. Sans aucun segment exploitable, repli sur le
    centre du bassin.
    """
    M = np.asarray(markers, dtype=np.float64)
    T = M.shape[0]
    nom = {n: i for i, n in enumerate(marker_names)}
    somme = np.zeros((T, 3))
    poids = np.zeros(T)
    for _, frac, prox, dist, pos in _SEGMENTS:
        ip = [nom[n] for n in prox if n in nom]
        if len(ip) != len(prox):
            continue
        p = _moyenne(M, ip)
        if dist is not None:
            idd = [nom[n] for n in dist if n in nom]
            if len(idd) != len(dist):
                continue
            p = p + pos * (_moyenne(M, idd) - p)
        ok = np.all(np.isfinite(p), axis=1)
        somme[ok] += frac * p[ok]
        poids[ok] += frac
    com = np.full((T, 3), np.nan)
    ok = poids > 0.5          # au moins la moitie de la masse est vue
    com[ok] = somme[ok] / poids[ok, None]
    ib = [nom[n] for n in _BASSIN if n in nom]
    if ib:
        b = _moyenne(M, ib)
        com[~ok] = b[~ok]
    return com


# ---------------------------------------------------------------------------
# Phases de contact
# ---------------------------------------------------------------------------


def _indices_pied(marker_names) -> dict[str, dict[str, list[int]]]:
    """Par cote : points plantaires (hauteur) et points du pied (centroide XZ).

    Hauteur : semelle SOLE_* si elle existe (points qui touchent vraiment le
    sol), sinon talon/orteil/5e meta cutanes. Position : toute la semelle et
    les reperes cutanes, dont le centroide est insensible a la bascule
    talon -> orteil (meme raison que dans anti_foot_skate_markers).
    """
    nom = {n: i for i, n in enumerate(marker_names)}
    out = {}
    for cote, suf in (("d", "_r"), ("g", "_l")):
        S = "R" if cote == "d" else "L"
        sole = [i for n, i in nom.items() if n.startswith("SOLE_") and n.endswith(suf)]
        peau = [nom[S + b] for b in ("CAL", "TOE", "MT5") if S + b in nom]
        out[cote] = {"hauteur": sorted(sole) if sole else peau,
                     "position": sorted(set(sole + peau))}
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _hysteresis(h: np.ndarray, entree: float, sortie: float) -> np.ndarray:
    out = np.zeros(h.size, dtype=bool)
    etat = False
    for t, v in enumerate(h):
        if not np.isfinite(v):
            etat = False
        elif etat:
            etat = v <= sortie
        else:
            etat = v < entree
        out[t] = etat
    return out


def _nettoyer(mask: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    """Recoud les trous INTERIEURS < min_off, puis jette les runs < min_on.

    Ordre de Mesh2Sim (_contacts_on_axis) : un appui fragmente par la gigue est
    d'abord recousu, ensuite seulement les miettes sont jetees. Les trous en
    bord de sequence sont laisses : un vrai vol en bord de clip ne doit pas
    etre comble par extrapolation.
    """
    m = np.asarray(mask, dtype=bool).copy()
    T = m.size
    for a, b in _runs(~m):
        if a > 0 and b < T and (b - a) < min_off:
            m[a:b] = True
    for a, b in _runs(m):
        if (b - a) < min_on:
            m[a:b] = False
    return m


def _parabole_g(y: np.ndarray, fps: float):
    """Ajuste y(t) = y0 + v0 t - g t^2 / 2 (g impose). Rend (ajuste, rms)."""
    n = y.size
    t = np.arange(n) / fps
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.full(n, np.nan), np.nan
    z = y + 0.5 * G * t ** 2
    A = np.column_stack([np.ones(ok.sum()), t[ok]])
    coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    fit = coef[0] + coef[1] * t - 0.5 * G * t ** 2
    return fit, float(np.sqrt(np.mean((fit[ok] - y[ok]) ** 2)))


def detecter_phases(
    markers: np.ndarray,
    marker_names,
    fps: float,
    *,
    seuil_entree_m: float = 0.03,
    seuil_sortie_m: float = 0.05,
    sol: str = "global",
    pct_sol: float = 10.0,
    fenetre_sol_s: float = 2.0,
    appui_min_s: float = 0.05,
    trou_max_s: float = 0.035,
    vol_min_s: float = 0.06,
    vol_max_s: float = 0.90,
    seuil_immobile_m: float = 0.10,
    v_immobile_ms: float = 0.25,
    immobile_min_s: float = 0.10,
    com: np.ndarray | None = None,
) -> dict:
    """Phases d'appui par pied et vols, par la HAUTEUR DES SEMELLES.

    Par pied : hauteur du point de semelle le plus bas au-dessus du sol ->
    hysteresis (on entre en contact plus bas qu'on n'en sort, sinon un pied
    pose qui oscille d'un millimetre autour du seuil bascule sans arret) ->
    trous courts recousus, appuis trop courts jetes.

    SOL. ``"global"`` : un percentile bas de la semelle la plus basse sur tout
    l'essai (sujet au sol la majeure partie du temps, comme ``jump.py``).
    ``"glissant"`` : enveloppe basse glissante sur ``fenetre_sol_s`` — pour un
    sol qui derive (camera portee), mais elle prend une marche/box pour le sol.

    PIED BAS ET IMMOBILE. Deuxieme voie vers le contact : semelle a moins de
    ``seuil_immobile_m`` du sol ET vitesse verticale sous ``v_immobile_ms``
    pendant au moins ``immobile_min_s``. Mesure sur ``outputs/NUIT_hop`` : a la
    reception du 2e bond, les semelles restent a 5-7 cm du sol pendant 0,3 s
    (le sujet s'enfonce, centre de masse 85 -> 61 cm) ; le seuil de hauteur
    seul prolongeait le vol jusque-la (0,80 s au lieu de 0,47). Un pied qui ne
    bouge plus verticalement, pres du sol, est pose.

    VOL = aucun pied en contact, entre ``vol_min_s`` et ``vol_max_s`` (au-dela,
    0,99 m de detente : pas un saut humain, c'est un passage en hauteur — le
    sujet debout sur une box). Deux gardes supplementaires, reprises de
    ``_detect_jumps_sole`` (synkro-analytics) :
      * un vol coupe par le bord du clip est marque ``bord`` (decollage ou
        reception non observes) ;
      * COHERENCE BALISTIQUE pour les vols >= 0,15 s : on ajuste une parabole
        LIBRE au centre de masse vertical sur le vol, et la gravite qu'elle
        implique doit etre comprise entre g/2 et 2g. Le faux vol type est une
        derive verticale lente de toute la scene (semelles ET bassin), dont
        la « gravite » est quasi nulle. ⚠️ Premiere version, abandonnee : le
        critere de ``_detect_jumps_sole`` (montee depuis le decollage entre la
        moitie et le double de g.T^2/8). Il suppose un vol SYMETRIQUE ; sur un
        hop pour la distance la reception se fait 25 cm plus bas que le
        decollage, la montee n'est que de 6 cm pour 16 « attendus », et un vrai
        vol balistique etait ecarte (NUIT_hop, 1,50 s).

    Returns:
        dict : ``contact`` {"d","g"} (T,) bool, ``appui`` (T,) bool (au moins
        un pied), ``hauteur`` {"d","g"} (T,) en metres, ``sol`` (T,) dans
        l'unite des marqueurs, ``vols`` liste de dicts {debut, fin, duree_s,
        bord, montee_m, prevu_m}, ``rejets`` liste de messages.
    """
    M = np.asarray(markers, dtype=np.float64)
    T = M.shape[0]
    u = _unite(M)
    idx = _indices_pied(marker_names)
    bas = {}
    for c in ("d", "g"):
        ih = idx[c]["hauteur"]
        if ih:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bas[c] = np.nanmin(M[:, ih, 1], axis=1)
    vide = {"contact": {}, "appui": np.zeros(T, bool), "hauteur": {},
            "sol": np.full(T, np.nan), "vols": [], "rejets": ["aucun point de pied"]}
    if not bas:
        return vide
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tous = np.nanmin(np.stack(list(bas.values()), axis=1), axis=1)
    fin = np.isfinite(tous)
    if fin.sum() < 3:
        return vide
    if sol == "glissant":
        from scipy.ndimage import percentile_filter
        w = max(3, int(round(fenetre_sol_s * fps)) | 1)
        tt = np.arange(T)
        rempli = np.interp(tt, tt[fin], tous[fin])
        niveau = percentile_filter(rempli, pct_sol, size=w, mode="nearest")
    else:
        niveau = np.full(T, float(np.nanpercentile(tous[fin], pct_sol)))

    min_on = max(2, int(round(appui_min_s * fps)))
    min_off = max(1, int(round(trou_max_s * fps)))
    contact, hauteur = {}, {}
    n_imm = max(2, int(round(immobile_min_s * fps)))
    for c, y in bas.items():
        h = (y - niveau) / u
        hauteur[c] = h
        m = _hysteresis(h, seuil_entree_m, seuil_sortie_m)
        # pied bas et immobile verticalement
        hf = _remplir(h)
        hl = np.convolve(np.pad(hf, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
        vz = np.gradient(hl) * fps
        imm = np.isfinite(h) & (h < seuil_immobile_m) & (np.abs(vz) < v_immobile_ms)
        imm = _nettoyer(imm, n_imm, 1)
        contact[c] = _nettoyer(m | imm, min_on, min_off)
    appui = np.zeros(T, bool)
    for m in contact.values():
        appui |= m
    # Une image sans aucune donnee de pied n'est ni appui ni vol : on ne
    # decrete pas un envol sur un trou de suivi.
    connu = np.zeros(T, bool)
    for y in bas.values():
        connu |= np.isfinite(y)

    if com is None:
        com = centre_de_masse_approx(M, marker_names)
    cy = com[:, 1] / u
    vols, rejets = [], []
    for a, b in _runs(~appui & connu):
        duree = (b - a) / fps
        bord = a == 0 or b >= T
        if duree < vol_min_s or (b - a) < 2:
            continue
        if duree > vol_max_s:
            rejets.append(f"passage en hauteur a {a / fps:.2f} s ({duree:.2f} s) : "
                          f"au-dela d'un vol humain ({vol_max_s:.2f} s), non corrige")
            continue
        info = {"debut": int(a), "fin": int(b), "duree_s": float(duree), "bord": bool(bord),
                "montee_m": None, "prevu_m": None}
        if not bord and duree >= 0.15:
            # duree du vol : b - a images en l'air, a la precision de l'image.
            seg = cy[a - 1:b + 1]
            ok = np.isfinite(seg)
            if ok.sum() >= 4:
                tt = np.arange(seg.size)[ok] / fps
                c2 = np.polyfit(tt, seg[ok], 2)[0]
                g_est = -2.0 * float(c2)
                info["g_estime"] = g_est
                info["montee_m"] = float(np.nanmax(seg) - seg[0]) if ok[0] else None
                info["prevu_m"] = G * duree ** 2 / 8.0
                if not (0.5 * G <= g_est <= 2.0 * G):
                    rejets.append(f"vol ecarte a {a / fps:.2f} s ({duree:.2f} s) : le centre "
                                  f"de masse tombe a {g_est:.1f} m/s2 au lieu de {G:.1f} "
                                  "(pas balistique)")
                    continue
        vols.append(info)
    return {"contact": contact, "appui": appui, "hauteur": hauteur, "sol": niveau,
            "vols": vols, "rejets": rejets}


# ---------------------------------------------------------------------------
# Moindres carres sur le decalage rigide
# ---------------------------------------------------------------------------


def _d2(T: int, centres, k: int = 1):
    """Lignes de difference seconde (pas k) centrees sur ``centres``."""
    import scipy.sparse as sp
    c = np.asarray([t for t in centres if t - k >= 0 and t + k < T], dtype=int)
    n = c.size
    rows = np.repeat(np.arange(n), 3)
    cols = np.column_stack([c - k, c, c + k]).ravel()
    vals = np.tile([1.0, -2.0, 1.0], n)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, T)), c


def _d1(T: int, centres, k: int = 1):
    """Lignes de difference centree (pas k) : x[t+k] - x[t-k]."""
    import scipy.sparse as sp
    c = np.asarray([t for t in centres if t - k >= 0 and t + k < T], dtype=int)
    n = c.size
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack([c - k, c + k]).ravel()
    vals = np.tile([-1.0, 1.0], n)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, T)), c


def _remplir(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return np.nan_to_num(x)
    t = np.arange(x.size)
    x[~ok] = np.interp(t[~ok], t[ok], x[ok])
    return x


def trajectoire_physique(
    markers: np.ndarray,
    marker_names,
    fps: float,
    *,
    phases: dict | None = None,
    corriger_vols: bool = True,
    parabole: bool = True,
    ancrer_appuis: bool = False,
    borner: bool = True,
    v_max_ms: float = 10.0,
    a_max_ms2: float = 15.0,
    echelle_bornes_s: float = 0.10,
    sigma_prior_m: float = 0.10,
    sigma_prior_appui_m: float = 0.02,
    sigma_lisse_ms2: float = 5.0,
    sigma_vol_ms2: float = 0.05,
    sigma_raccord_ms2: float = 8.0,
    sigma_appui_m: float = 0.02,
    sigma_appui_vertical_m: float = 0.003,
    sigma_glisse_appui_ms: float = 0.15,
    max_shift_m: float = 0.50,
    max_ecart_vol_m: float = 0.25,
    vols_en_bord: bool = False,
    conserver_distance_vol: bool = True,
    iterations_bornes: int = 8,
    **kw_phases,
):
    """Decalage rigide qui rend la trajectoire du centre de masse physique.

    Resout, axe par axe, le probleme de moindres carres sur s (T,) :

        Somme_t (s_t / sigma_prior_t)^2                     correction petite
              (sigma_prior_appui en appui, plus serre)
      + Somme_t (D2 s_t / (sigma_lisse dt^2))^2             correction lisse (C1)
      + Somme_vols (D2 (c+s)_t - a_vol dt^2)^2 / (sigma_vol dt^2)^2
              a_vol = 0 en X et Z, -g en Y                 mecanique du vol
      + Somme_raccords (D2 (c+s)_t / (sigma_raccord dt^2))^2
              au decollage et a la reception : la vitesse   continuite C1
              ne saute pas, elle ne change que sous une force
      + [X,Z] Somme_vols ((s_reception - s_decollage) / 1 mm)^2
                                                            distance de vol conservee
      + [X,Z] Somme_appui ((s_t+1 - s_t) / (sigma_glisse dt))^2
                                                            la correction ne fait pas
                                                            patiner le pied pose
      + [Y] Somme_appui (s_t / sigma_appui_vertical)^2      le pied pose reste au sol
      + [option] variance du pied pendant chaque appui      pas de glissement
      + [bornes] la ou la trajectoire corrigee depasse       vitesses humaines
              a_max : (a / sigma)^2 -> 0, poids x4 a chaque passe tant que
              ca depasse (Whittaker local) ; v_max : (v - v_ecretee)^2

    c = centre de masse approche, D2 = difference seconde, dt = 1/fps.

    Les vols dont le decollage ou la reception est hors du clip (``bord``)
    sont ignores par defaut ; avec ``vols_en_bord=True`` ils recoivent la
    contrainte horizontale seulement (la parabole n'a pas de point d'appui
    d'un cote, on ne l'invente pas).

    Args:
        markers: (T, N, 3), Y vers le haut, m ou mm.
        marker_names: N noms (markerset Flodelaplace, semelles SOLE_*).
        fps: frequence d'image.
        phases: resultat de :func:`detecter_phases` (recalcule sinon, avec
            ``kw_phases``).
        corriger_vols / parabole / ancrer_appuis / borner: interrupteurs.
        v_max_ms / a_max_ms2: bornes de la vitesse et de l'acceleration
            HORIZONTALES du centre de masse, mesurees a l'echelle
            ``echelle_bornes_s`` (10 cm de gigue image a image n'est pas une
            acceleration du corps ; la borne vise les deplacements sur 0,1 s).
        max_shift_m: garde-fou. Au-dela, ce n'est plus une correction mais
            une perte de suivi : l'etage rend les marqueurs INCHANGES.

    Returns:
        (markers_corriges, decalages, rapport)
        decalages : (T, 3) en METRES, a ajouter tel quel au mesh (X, Y, Z).
        rapport : dict (phases, vols corriges, bornes actives, avertissements).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    M = np.asarray(markers, dtype=np.float64)
    rapport: dict = {"applique": False, "avertissements": []}
    if M.ndim != 3 or M.shape[0] < 5 or M.shape[2] != 3:
        rapport["avertissements"].append("sequence trop courte ou forme invalide")
        return markers, None, rapport
    T = M.shape[0]
    u = _unite(M)
    dt = 1.0 / fps
    com = centre_de_masse_approx(M, marker_names) / u          # metres
    if np.isfinite(com).all(axis=1).sum() < 5:
        rapport["avertissements"].append("centre de masse introuvable")
        return markers, None, rapport
    if phases is None:
        phases = detecter_phases(M, marker_names, fps, com=com * u, **kw_phases)
    rapport["phases"] = {"vols": phases["vols"], "rejets": phases["rejets"],
                         "part_appui": float(phases["appui"].mean())}
    vols = list(phases["vols"]) if corriger_vols else []
    appui = phases["appui"]
    # VOLS EN BORD DE CLIP : ecartes par defaut. Decollage ou reception hors
    # champ, on ne sait pas ou la droite commence ; et mesure sur DIAG_run, un
    # clip qui commence semelles en l'air est bien plus souvent un sol mal
    # pose (0,54 s « en vol » a la premiere image) qu'un vrai vol.
    if not vols_en_bord:
        vols = [v for v in vols if not v["bord"]]
    # GARDE PAR VOL : une trajectoire qui s'ecarte de plus de `max_ecart_vol_m`
    # de la droite decollage -> reception n'est pas une erreur de profondeur a
    # redresser mais un saut de suivi (SOL2_basket, 40 cm en 0,13 s). La
    # redresser demanderait une correction enorme qui deborderait sur les
    # appuis voisins : on laisse ce vol tel quel et on le signale.
    _gardes = []
    for v in vols:
        a, b = v["debut"], v["fin"]
        if v["bord"]:
            _gardes.append(v)
            continue
        seg = com[a - 1:b + 1][:, [0, 2]]
        if not np.isfinite(seg).all():
            _gardes.append(v)
            continue
        lam = np.linspace(0.0, 1.0, seg.shape[0])[:, None]
        ecart = float(np.linalg.norm(seg - (seg[0] + lam * (seg[-1] - seg[0])), axis=1).max())
        if ecart > max_ecart_vol_m:
            rapport["avertissements"].append(
                f"vol a {a / fps:.2f} s laisse tel quel : ecart de {ecart * 100:.0f} cm a la "
                f"droite de vol (> {max_ecart_vol_m * 100:.0f} cm, saut de suivi)")
            continue
        _gardes.append(v)
    vols = _gardes
    rapport["vols_corriges"] = [(v["debut"], v["fin"]) for v in vols]

    # Centres des contraintes de vol : de la premiere a la derniere image en
    # l'air, la difference seconde en t engage t-1 et t+1, donc les images
    # d'appui qui encadrent le vol (decollage et reception). Pour un vol en
    # bord de clip, seules les differences qui restent dans le clip.
    c_vol, c_vol_complet, c_raccord = [], [], []
    for v in vols:
        a, b = v["debut"], v["fin"]
        cs = list(range(max(a, 1), min(b, T - 1)))
        c_vol += cs
        if not v["bord"]:
            c_vol_complet += cs
            c_raccord += [a - 1, b]
    c_vol = sorted(set(c_vol))

    # Pieds (variance d'appui, option)
    idx_pieds = _indices_pied(marker_names)
    pieds_xz = {}
    for c in ("d", "g"):
        ip = idx_pieds[c]["position"]
        if ip:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pieds_xz[c] = np.nanmean(M[:, ip, :][:, :, [0, 2]], axis=1) / u

    def _bloc_variance(axe: int):
        """Normal equations de la variance d'appui : (A, b) avec s^T A s + 2 b^T s."""
        rr, cc, vv = [], [], []
        bb = np.zeros(T)
        for c, contact in phases["contact"].items():
            if c not in pieds_xz:
                continue
            x = pieds_xz[c][:, axe]
            for a_, b_ in _runs(contact):
                seg = np.arange(a_, b_)
                ok = np.isfinite(x[seg])
                seg = seg[ok]
                W = seg.size
                if W < 3:
                    continue
                P = (np.eye(W) - np.ones((W, W)) / W) / sigma_appui_m ** 2
                I, J = np.meshgrid(seg, seg, indexing="ij")
                rr.append(I.ravel()); cc.append(J.ravel()); vv.append(P.ravel())
                bb[seg] += P @ x[seg]
        if not rr:
            return sp.csr_matrix((T, T)), bb
        A = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                          shape=(T, T)).tocsr()          # doublons sommes
        return A, bb

    k_b = max(1, int(round(0.5 * echelle_bornes_s * fps)))
    tous = np.arange(T)
    D2_lisse, c_l = _d2(T, tous)
    D2_vol, c_v = _d2(T, c_vol)
    D2_vol_y, c_vy = _d2(T, sorted(set(c_vol_complet)))
    D2_rac, c_r = _d2(T, sorted(set(c_raccord)))
    # DISTANCE DE VOL CONSERVEE : s(reception) = s(decollage) en X et Z. Les
    # deux images d'appui qui encadrent le vol sont les mieux observees, et la
    # distance qui les separe EST la mesure clinique du hop (et le
    # deplacement d'un saut). Sans cette egalite, la correction de forme du vol
    # fuyait sur ses extremites : NUIT_hop, 2e bond, 114,6 -> 109,5 cm sans
    # bornes et 96,4 cm avec. On redresse la forme, jamais la distance.
    _dv = [(v["debut"] - 1, v["fin"]) for v in vols if not v["bord"] and v["fin"] < T]
    D_dist = sp.csr_matrix((np.tile([-1.0, 1.0], len(_dv)),
                            (np.repeat(np.arange(len(_dv)), 2),
                             np.asarray(_dv, dtype=int).ravel() if _dv else np.zeros(0, int))),
                           shape=(len(_dv), T))

    # LA CORRECTION N'AJOUTE PAS DE GLISSEMENT. Entre deux images d'appui
    # consecutives, le decalage horizontal ne doit presque pas varier : sinon
    # c'est l'etage lui-meme qui fait patiner le pied pose (mesure sur
    # NUIT_jump sans ce terme : le raccord C1 de la reception se relachait sur
    # l'appui suivant, 4 cm de glissement ajoute). Difference AVANT (s[t+1] -
    # s[t]) sur les paires ou les deux images sont en appui.
    # 0,15 m/s ET NON 0,05 : trop strict, le decalage ne peut plus se relacher
    # nulle part entre deux bonds et les corrections de vol s'ADDITIONNENT.
    # Mesure sur NUIT_hop (3 bonds) : deplacement net 4,10 -> 3,83 m a 0,05,
    # 4,09 m a 0,15, glissement de mi-appui P90 12,9 -> 11,8 cm.
    _pa = np.flatnonzero(appui[:-1] & appui[1:])
    D1_appui = sp.csr_matrix((np.tile([-1.0, 1.0], _pa.size),
                              (np.repeat(np.arange(_pa.size), 2),
                               np.column_stack([_pa, _pa + 1]).ravel())),
                             shape=(_pa.size, T))

    def _resoudre(axe: int, extra_rows=None) -> np.ndarray:
        """Resout pour l'axe (0 = X, 1 = Y, 2 = Z). extra_rows : [(D, cible, sigma)]."""
        x = _remplir(com[:, axe])
        # Rappel vers zero plus FORT en appui : la position d'un pied pose est
        # la mieux observee (et deja ancree par l'anti-glissement), c'est le
        # vol qui porte l'erreur. Avec un rappel uniforme, le clip entier
        # absorbait une part de la bosse de profondeur du vol : 1,8 cm de
        # decalage sur toute la station debout d'un saut synthetique.
        sig = np.where(appui, sigma_prior_appui_m, sigma_prior_m)
        H = sp.diags(1.0 / sig ** 2).tocsr()
        g = np.zeros(T)
        wl = 1.0 / (sigma_lisse_ms2 * dt * dt) ** 2
        H = H + wl * (D2_lisse.T @ D2_lisse)
        if axe == 1:
            if parabole and D2_vol_y.shape[0]:
                wv = 1.0 / (sigma_vol_ms2 * dt * dt) ** 2
                r0 = D2_vol_y @ x + G * dt * dt           # D2(c) - (-g dt^2)
                H = H + wv * (D2_vol_y.T @ D2_vol_y)
                g = g + wv * (D2_vol_y.T @ r0)
            # Le pied pose reste au sol : pas de decalage vertical en appui.
            wa = np.where(appui, 1.0 / sigma_appui_vertical_m ** 2, 0.0)
            H = H + sp.diags(wa)
        else:
            if D2_vol.shape[0]:
                wv = 1.0 / (sigma_vol_ms2 * dt * dt) ** 2
                H = H + wv * (D2_vol.T @ D2_vol)
                g = g + wv * (D2_vol.T @ (D2_vol @ x))
            if D2_rac.shape[0]:
                wr = 1.0 / (sigma_raccord_ms2 * dt * dt) ** 2
                H = H + wr * (D2_rac.T @ D2_rac)
                g = g + wr * (D2_rac.T @ (D2_rac @ x))
            if conserver_distance_vol and D_dist.shape[0]:
                H = H + (1.0 / 0.001 ** 2) * (D_dist.T @ D_dist)      # 1 mm
            if D1_appui.shape[0]:
                wg = 1.0 / (sigma_glisse_appui_ms * dt) ** 2
                H = H + wg * (D1_appui.T @ D1_appui)
            if ancrer_appuis:
                ai = 0 if axe == 0 else 1
                A, bb = _bloc_variance(ai)
                H = H + A
                g = g + bb
            for D, cible, sig in (extra_rows or []):
                W = sp.diags(1.0 / np.asarray(sig, dtype=np.float64) ** 2)
                H = H + D.T @ W @ D
                g = g + D.T @ (W @ (D @ x - cible))
        return spsolve(H.tocsc(), -g)

    s = np.zeros((T, 3))
    if corriger_vols or ancrer_appuis:
        s[:, 0] = _resoudre(0)
        s[:, 2] = _resoudre(2)
    if corriger_vols and parabole:
        s[:, 1] = _resoudre(1)

    s_sans_bornes = s.copy()

    # ── Bornes de vitesse et d'acceleration horizontales ──────────────────
    # Non lineaires (norme horizontale) : on repere les images ou la
    # trajectoire CORRIGEE depasse et on resout de nouveau avec une contrainte
    # locale, jusqu'a ce qu'il n'y ait plus de depassement.
    #
    # * ACCELERATION : contrainte D2(c+s) -> 0 a l'image fautive, dont le poids
    #   quadruple tant qu'elle depasse — un lissage de Whittaker LOCAL. Premiere
    #   version, abandonnee : viser l'acceleration ecretee (meme direction,
    #   norme a la borne). Sur un signal qui tremble, les cibles ecretees
    #   s'integrent en une derive : FINAL_basket se deplacait de 15 cm de plus
    #   a l'arrivee. Viser zero laisse la tendance intacte (une droite est dans
    #   le noyau de D2), donc le deplacement net.
    # * VITESSE : cible = vitesse ecretee dans la meme direction (viser zero
    #   arreterait le sujet). Rarement active : 10 m/s par defaut.
    actives_v, actives_a = set(), set()

    def _borner(s0):
        s = s0.copy()
        cibles_v: dict[int, np.ndarray] = {}
        poids_a: dict[int, float] = {}
        for _ in range(max(0, iterations_bornes)):
            q = np.column_stack([_remplir(com[:, 0]), _remplir(com[:, 2])]) + s[:, [0, 2]]
            Dv, cv = _d1(T, tous, k_b)
            Da, ca = _d2(T, tous, k_b)
            vel = (Dv @ q) / (2 * k_b * dt)
            acc = (Da @ q) / (k_b * dt) ** 2
            nv = np.linalg.norm(vel, axis=1)
            na = np.linalg.norm(acc, axis=1)
            nouveau = False
            for i in np.flatnonzero(nv > v_max_ms * 1.01):
                cibles_v[int(cv[i])] = vel[i] * (v_max_ms / nv[i])
                nouveau = True
            for i in np.flatnonzero(na > a_max_ms2 * 1.01):
                t = int(ca[i])
                poids_a[t] = min(poids_a.get(t, 0.25) * 4.0, 4.0 ** 6)
                nouveau = True
            if not nouveau:
                break
            actives_v.update(cibles_v)
            actives_a.update(poids_a)
            for axe, j in ((0, 0), (2, 1)):
                extra = []
                if cibles_v:
                    D, _c = _d1(T, sorted(cibles_v), k_b)
                    cib = np.array([cibles_v[t][j] for t in _c]) * (2 * k_b * dt)
                    extra.append((D, cib, np.full(_c.size, 0.05 * 2 * k_b * dt)))
                if poids_a:
                    D, _c = _d2(T, sorted(poids_a), k_b)
                    sig = a_max_ms2 * (k_b * dt) ** 2 / np.sqrt([poids_a[t] for t in _c])
                    extra.append((D, np.zeros(_c.size), sig))
                s[:, axe] = _resoudre(axe, extra)
        return s

    def _depasse(s):
        return (not np.all(np.isfinite(s))) or float(np.linalg.norm(s, axis=1).max()) > max_shift_m

    if borner:
        s = _borner(s)
        rapport["bornes"] = {"vitesse_images": len(actives_v),
                             "acceleration_images": len(actives_a)}
        # Repli : si les bornes font deborder le garde-fou (trajectoire qui
        # saute partout, le lissage n'en finit plus), on garde au moins la
        # correction des vols, sans bornes.
        if _depasse(s) and not _depasse(s_sans_bornes):
            rapport["avertissements"].append(
                "bornes abandonnees : elles demandaient plus que le garde-fou "
                f"({max_shift_m * 100:.0f} cm), trajectoire trop bruitee ; vols seuls")
            rapport["bornes"]["abandonnees"] = True
            s = s_sans_bornes

    n = np.linalg.norm(s, axis=1)
    if not np.all(np.isfinite(s)):
        rapport["avertissements"].append("solution non finie : marqueurs inchanges")
        return markers, None, rapport
    if float(n.max()) > max_shift_m:
        rapport["avertissements"].append(
            f"correction de {n.max() * 100:.0f} cm > garde-fou {max_shift_m * 100:.0f} cm : "
            "perte de suivi probable, marqueurs inchanges")
        return markers, None, rapport

    out = M + (s * u)[:, None, :]
    rapport["applique"] = True
    rapport["correction_max_cm"] = float(n.max() * 100)
    rapport["correction_max_horizontale_cm"] = float(np.linalg.norm(s[:, [0, 2]], axis=1).max() * 100)
    rapport["correction_max_verticale_cm"] = float(np.abs(s[:, 1]).max() * 100)
    return out.astype(np.asarray(markers).dtype, copy=False), s, rapport


def appliquer_decalages(points, decalages):
    """Rejoue les decalages (T, 3) en METRES sur des points en METRES.

    ``points`` : tableau (T, K, 3) ou liste de T tableaux (K, 3) / None — la
    forme des ``verts_world`` / ``kpts_world`` / ``jc_world`` de
    demo_video_opensim.py. Longueurs differentes : decalages reechantillonnes
    lineairement (meme regle que pour ``_lateral_shifts``). Modifie une COPIE.
    """
    if decalages is None:
        return points
    d = np.asarray(decalages, dtype=np.float64)
    n = len(points)
    if len(d) != n:
        src = np.linspace(0.0, 1.0, len(d))
        dst = np.linspace(0.0, 1.0, n)
        d = np.stack([np.interp(dst, src, d[:, k]) for k in range(3)], axis=1)
    if isinstance(points, np.ndarray):
        return points + d[:, None, :].astype(points.dtype)
    return [None if p is None else np.asarray(p) + d[i][None, :].astype(np.asarray(p).dtype)
            for i, p in enumerate(points)]


# ---------------------------------------------------------------------------
# Metriques (evaluation hors ligne)
# ---------------------------------------------------------------------------


def metriques_trajectoire(markers: np.ndarray, marker_names, fps: float,
                          phases: dict, echelle_s: float = 0.10) -> dict:
    """Chiffres de coherence physique d'une trajectoire, sur des phases FIXEES.

    Les phases doivent etre les memes avant et apres (on juge la trajectoire,
    pas la detection) — une translation verticale en vol ne change d'ailleurs
    pas l'etat des semelles tant qu'elle ne les ramene pas sous le seuil.

    * vols : ecart maximal du centre de masse horizontal a la droite
      decollage -> reception (cm, par axe : X = profondeur camera, Z = lateral
      image), deplacement horizontal pendant le vol, saut de vitesse
      horizontale au decollage (m/s), RMS a la parabole g ajustee (cm) ;
    * appuis : glissement du pied, ecart maximal a sa position mediane par
      appui (cm) ;
    * global : deplacement net du bassin (m), vitesse/acceleration
      horizontales du centre de masse a l'echelle ``echelle_s`` (P99, max).
    """
    M = np.asarray(markers, dtype=np.float64)
    T = M.shape[0]
    u = _unite(M)
    com = centre_de_masse_approx(M, marker_names) / u
    out: dict = {"vols": []}
    for v in phases["vols"]:
        a, b = v["debut"], v["fin"]
        if v["bord"]:
            continue
        seg = com[a - 1:b + 1]
        if not np.isfinite(seg).all():
            continue
        n = seg.shape[0]
        lam = np.linspace(0.0, 1.0, n)[:, None]
        droite = seg[0] + lam * (seg[-1] - seg[0])
        ecart = seg - droite
        _, rms = _parabole_g(seg[:, 1], fps)
        # saut de vitesse horizontale au decollage : vitesse sur les 2 images
        # d'appui avant, contre vitesse moyenne du vol
        vj = np.nan
        if a - 3 >= 0:
            v_av = (com[a - 1, [0, 2]] - com[a - 3, [0, 2]]) / (2.0 / fps)
            v_vol = (seg[-1, [0, 2]] - seg[0, [0, 2]]) / ((n - 1) / fps)
            vj = float(np.linalg.norm(v_vol - v_av))
        out["vols"].append({
            "debut_s": a / fps, "duree_s": v["duree_s"],
            "ecart_x_cm": float(np.abs(ecart[:, 0]).max() * 100),
            "ecart_z_cm": float(np.abs(ecart[:, 2]).max() * 100),
            "depl_h_cm": float(np.linalg.norm(seg[-1, [0, 2]] - seg[0, [0, 2]]) * 100),
            "depl_x_cm": float((seg[-1, 0] - seg[0, 0]) * 100),
            "saut_vitesse_ms": vj,
            "rms_parabole_cm": float(rms * 100),
            "montee_cm": float((np.nanmax(seg[:, 1]) - seg[0, 1]) * 100),
            "prevu_cm": float(G * v["duree_s"] ** 2 / 8.0 * 100),
        })
    # GLISSEMENT : mesure sur le MILIEU de chaque appui (50 % central). Les
    # bords portent l'attaque et le deroule, ou le centroide du pied bouge
    # reellement (bascule talon -> orteil) ; et un « appui » detecte par la
    # hauteur peut fusionner deux pas traines. Ecart maximal du centroide XZ a
    # sa mediane, plus la vitesse horizontale mediane du pied en appui.
    idx = _indices_pied(marker_names)
    gl, vit = [], []
    for c, contact in phases["contact"].items():
        ip = idx[c]["position"]
        if not ip:
            continue
        with np.errstate(all="ignore"):
            xz = np.nanmean(M[:, ip, :][:, :, [0, 2]], axis=1) / u
        for a_, b_ in _runs(contact):
            L = b_ - a_
            if L < 6:
                continue
            p = xz[a_ + L // 4:b_ - L // 4]
            p = p[np.all(np.isfinite(p), axis=1)]
            if len(p) < 3:
                continue
            gl.append(float(np.max(np.linalg.norm(p - np.median(p, axis=0), axis=1)) * 100))
            vit.append(float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)) * fps))
    out["glissement_cm"] = {"n": len(gl),
                            "mediane": float(np.median(gl)) if gl else np.nan,
                            "p90": float(np.percentile(gl, 90)) if gl else np.nan,
                            "vitesse_mediane_ms": float(np.median(vit)) if vit else np.nan}
    # DEPLACEMENT NET : medianes du bassin sur les premiers et derniers 10 %
    # (au moins 0,3 s). Une moyenne des 5 premieres images tombait en plein
    # saut de suivi sur FINAL_basket (bassin +50 cm puis retour en 0,2 s) et
    # faisait passer un lissage legitime pour un changement de deplacement.
    nom = {n: i for i, n in enumerate(marker_names)}
    ib = [nom[n] for n in _BASSIN if n in nom]
    if ib:
        b = _moyenne(M, ib)[:, [0, 2]] / u
        k = int(min(max(0.3 * fps, 0.1 * T), T // 2)) or 1
        d = np.nanmedian(b[-k:], 0) - np.nanmedian(b[:k], 0)
        out["deplacement_net_m"] = float(np.linalg.norm(d))
        out["deplacement_net_xz_m"] = d.tolist()
    kb = max(1, int(round(0.5 * echelle_s * fps)))
    q = np.column_stack([_remplir(com[:, 0]), _remplir(com[:, 2])])
    if T > 2 * kb + 1:
        vel = (q[2 * kb:] - q[:-2 * kb]) / (2 * kb / fps)
        acc = (q[2 * kb:] - 2 * q[kb:-kb] + q[:-2 * kb]) / (kb / fps) ** 2
        nv, na = np.linalg.norm(vel, axis=1), np.linalg.norm(acc, axis=1)
        out["vitesse_h"] = {"p99": float(np.percentile(nv, 99)), "max": float(nv.max())}
        out["acceleration_h"] = {"p99": float(np.percentile(na, 99)), "max": float(na.max())}
    return out
