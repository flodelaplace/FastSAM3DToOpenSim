"""Audit des reglages par module et sous-mode.

Ne RECOPIE PAS la logique : extrait les VRAIES lignes de decision de
`demo_video_opensim.py` et les execute sur un jeu d'arguments simule. Un audit
qui recopierait les regles pourrait passer au vert pendant que le code diverge.
"""
import argparse, io, os, re, sys, contextlib
sys.path.insert(0, "/home/fdela/FastSAM3DToOpenSim")
SRC = open("/home/fdela/FastSAM3DToOpenSim/demo_video_opensim.py").read().split("\n")

def bloc(premiere, derniere):
    """Lignes [premiere, derniere] du fichier (1-index), desindentees de 4."""
    return "\n".join(l[4:] if l.startswith("    ") else l
                     for l in SRC[premiere - 1:derniere])

def borne(motif, apres=0):
    for k, l in enumerate(SRC):
        if k + 1 > apres and re.match(motif, l):
            return k + 1
    raise SystemExit(f"introuvable: {motif}")

L_LAT   = borne(r"    _auto_lock_lateral = ")
L_STEFF = borne(r"    _stationary_effective = ")
L_CLAMP = borne(r'    os\.environ\["NO_FLOOR_CLAMP"\]')
L_SF    = borne(r"    _sf_exclu = ")
L_SFEND = borne(r"    _stable_floor = ")
L_AS    = borne(r"    _anti_skate_on = ")
L_ASEND = borne(r"            _anti_skate_on = False")
L_EP    = borne(r"    _en_place = ")
L_TRAJ  = borne(r"    _traj_lisse = ")
L_FA    = borne(r"    if _auto_feet_anchor and not args.feet_anchor")

MORCEAUX = [bloc(L_LAT, L_STEFF), bloc(L_CLAMP, L_CLAMP + 4),
            bloc(L_SF, L_SFEND), bloc(L_AS, L_ASEND),
            bloc(L_EP, L_EP + 1), bloc(L_TRAJ, L_TRAJ + 1),
            bloc(L_FA, L_FA + 4)]   # suppression effective de feet_anchor

DEFAUTS = dict(module=None, stationary=False, handheld=False, danseuse=False,
               treadmill_speed=None, lock_vertical=False, lock_lateral=False,
               feet_anchor=False, contact_anchor=False, no_anti_foot_skate=False,
               stable_floor=False, no_stable_floor=False, no_traj_smooth=False,
               anti_skate_variance=False, floor=False, floor_seated=False,
               floor_moge=False, no_floor_moge=False, no_lean_fix=False,
               no_world_frame=False, lateral_anchor=False, no_lateral_anchor=False,
               cycling_position="road", hop_type=None, leg=None)

def regler(**kw):
    a = argparse.Namespace(**{**DEFAUTS, **kw})
    ns = {"args": a, "os": os, "getattr": getattr}
    with contextlib.redirect_stdout(io.StringIO()):
        for m in MORCEAUX:
            exec(m, ns)
    return a, ns

CAS = [
    ("squat",              dict(module="d3.squat")),
    ("lever de chaise",    dict(module="d3.sit_to_stand")),
    ("squat unipodal",     dict(module="d3.single_leg_squat")),
    ("saut vertical",      dict(module="d3.jump")),
    ("hop unipodal",       dict(module="d3.single_leg_hop")),
    ("marche",             dict(module="d3.gait")),
    ("marche tapis",       dict(module="d3.gait", treadmill_speed=1.2)),
    ("marche cam portee",  dict(module="d3.gait", handheld=True)),
    ("course",             dict(module="d3.running")),
    ("course tapis",       dict(module="d3.running", treadmill_speed=3.0)),
    ("course cam portee",  dict(module="d3.running", handheld=True)),
    ("depart sprint",      dict(module="d3.sprint_start")),
    ("cyclisme route",     dict(module="d3.cycling")),
    ("cyclisme danseuse",  dict(module="d3.cycling", danseuse=True)),
    ("cyclisme cam portee",dict(module="d3.cycling", handheld=True)),
]
COLS = [("stationnaire", "_stationary_effective"), ("verrou vertical", "_auto_lock_vertical"),
        ("verrou lateral", "_auto_lock_lateral"), ("anti-glisse", "_anti_skate_on"),
        ("en place", "_en_place"), ("traj lissee", "_traj_lisse"),
        ("sol stable", "_stable_floor"), ("ancrage pieds", "_auto_feet_anchor"),
        ("ancre contact", "_auto_contact_anchor")]
print(f"{'cas':22s} " + " ".join(f"{n:>15s}" for n, _ in COLS) + "   clamp sol")
for nom, kw in CAS:
    a, ns = regler(**kw)
    vals = []
    for _, v in COLS:
        x = ns.get(v)
        if v == "_auto_lock_vertical":
            x = x or a.lock_vertical
        if v == "_auto_contact_anchor" and ns.get("_stable_floor"):
            # `transform()` fait `if stable_floor: ... elif contact_anchor:` :
            # le sol stable gagne, l'ancrage conscient du contact n'est JAMAIS
            # atteint. Le signaler plutot que d'afficher un « oui » trompeur.
            vals.append("MORT"); continue
        vals.append("oui" if x else "non")
    # Le clamp de penetration vit dans `elif correct_floor_lean:` de
    # `transform()`, donc il est INATTEIGNABLE des que le sol stable est actif.
    # Afficher « oui » la ou il ne s'execute jamais serait un faux positif.
    if ns.get("_stable_floor"):
        clamp = "MORT"
    else:
        clamp = "coupe" if os.environ.get("NO_FLOOR_CLAMP") == "1" else "oui"
    print(f"{nom:22s} " + " ".join(f"{v:>15s}" for v in vals) + f"   {clamp}")
