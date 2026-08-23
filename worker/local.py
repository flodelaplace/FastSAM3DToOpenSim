"""Enchaîne plusieurs vidéos dans UN SEUL process, sans SQS ni AWS.

C'est le banc d'essai du worker persistant. Il sert à une chose : prouver
qu'une vidéo n'en contamine pas une autre. Deux fuites d'état rendaient ça
faux jusqu'au 2026-08-21 — `NO_FLOOR_CLAMP` posé par une vidéo de course et
jamais retiré, et le tracker BoT-SORT qui restait activé.

    python -m worker.local --job "--video_path a.mp4 --output_dir /outputs/a ..." \
                           --job "--video_path b.mp4 --output_dir /outputs/b ..."

Chaque --job est une ligne de commande complète de demo_video_opensim.py.
Combinée à --deterministic, la comparaison attendue est l'égalité OCTET PAR
OCTET avec les mêmes vidéos passées une par une.
"""
import argparse
import shlex
import sys
import time

from worker.session import InferenceSession


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", action="append", required=True,
                   help="Une ligne de commande demo_video_opensim.py, entre guillemets. "
                        "Répétable : les jobs s'exécutent dans l'ordre donné.")
    p.add_argument("--detector", default="yolo_pose")
    p.add_argument("--detector_model",
                   default="checkpoints/yolo/yolo11m-pose.engine")
    p.add_argument("--local_checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    args = p.parse_args()

    # Le déterminisme doit être armé AVANT que CUDA s'initialise, donc avant le
    # chargement des modèles. Dans le CLI c'est automatique (main() s'exécute en
    # premier) ; ici les modèles se chargent d'abord, il faut donc l'anticiper.
    if any("--deterministic" in shlex.split(j) for j in args.job):
        import demo_video_opensim as pipeline
        pipeline._activer_determinisme()

    t0 = time.time()
    session = InferenceSession(detector=args.detector,
                               detector_model=args.detector_model,
                               local_checkpoint=args.local_checkpoint)
    t_charge = time.time() - t0
    print(f"\n[worker] modèles chargés en {t_charge:.1f} s "
          f"— ce coût n'est payé qu'une fois\n", flush=True)

    echecs = 0
    for i, ligne in enumerate(args.job, 1):
        argv = shlex.split(ligne)
        print(f"\n{'=' * 70}\n[worker] job {i}/{len(args.job)}\n{'=' * 70}", flush=True)
        t = time.time()
        try:
            session.run(argv)
            print(f"[worker] job {i} terminé en {time.time() - t:.1f} s", flush=True)
        except BaseException as err:                       # noqa: BLE001
            # SystemExit n'hérite pas d'Exception : un `except Exception` le
            # laisserait tuer le worker. Une vidéo qui échoue ne doit jamais
            # emporter les suivantes.
            if isinstance(err, KeyboardInterrupt):
                raise
            echecs += 1
            print(f"[worker] job {i} EN ÉCHEC après {time.time() - t:.1f} s : "
                  f"{type(err).__name__}: {err}", flush=True)
        alloue, reserve = session.gpu_memory()
        print(f"[worker] GPU alloué {alloue:.0f} Mo | réservé {reserve:.0f} Mo "
              f"| {session.jobs_done} vidéos traitées", flush=True)

    print(f"\n[worker] {len(args.job)} jobs, {echecs} échec(s), "
          f"{time.time() - t0:.1f} s au total "
          f"(dont {t_charge:.1f} s de chargement, payés une fois)")
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
