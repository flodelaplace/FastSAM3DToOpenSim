"""Worker persistant : charge les modèles une fois, consomme une file SQS.

Pourquoi : chaque vidéo repayait 128 s de re-chauffage (sync des checkpoints,
imports Python, chargement des modèles et warmup TensorRT) sur 257 s de job.
Ici c'est payé une fois au démarrage, puis chaque vidéo ne coûte que son calcul.

Le worker s'éteint tout seul après IDLE_EXIT_SECONDS sans message. AWS Batch
libère alors l'instance. Toute la politique de coût tient donc dans ce nombre :
court = on paie peu mais on redémarre souvent, long = l'inverse. Pas besoin de
retoucher l'infrastructure pour l'ajuster.

Format d'un message (JSON) :
    {"s3_input": "s3://…/video.mp4", "s3_output": "s3://…/prefixe/",
     "extra_args": "--person_height 1.80 --floor_moge",
     "trim_start": "3", "trim_end": "12",           (facultatifs)
     "inference_type": "body"}                       (facultatif)
"""
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

QUEUE_URL = os.environ["WORK_QUEUE_URL"]
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
IDLE_EXIT_SECONDS = int(os.environ.get("IDLE_EXIT_SECONDS", "600"))
MAX_JOBS = int(os.environ.get("MAX_JOBS_PER_WORKER", "20"))
MAX_AGE_SECONDS = int(os.environ.get("MAX_WORKER_AGE_SECONDS", "14400"))
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "60"))
VISIBILITY_SECONDS = int(os.environ.get("VISIBILITY_SECONDS", "1200"))
# BUDGET D'IMAGES PAR VIDEO. Le post-traitement garde le maillage de chaque
# image en memoire : le 2026-09-24, un geste libre de 3 min 19 (5 949 images)
# a tue le worker (OutOfMemoryError, 14 Go) apres 20 min de calcul, et le
# message, remis en file, aurait tue les workers suivants un par un. Au-dela
# du budget : on descend d'abord a 30 images/s, puis on ne traite que le debut
# de la video, et on le DIT (mail + analysis_note.json dans la sortie).
# Mesure locale sur cette video (2026-09-24) : pic 12,75 Go a 3 000 images,
# dont ~7,2 Go fixes (modeles + inference) et ~1,8 Mo par image au
# post-traitement. Le worker a 14 Go et garde un residu d'une video a
# l'autre : 1 800 images (60 s a 30 images/s) visent ~10,5 Go de pic.
MAX_FRAMES = int(os.environ.get("MAX_FRAMES_PER_VIDEO", "1800"))
FPS_REPLI = 30.0

# Code de sortie reserve aux etats CUDA irrecuperables : Batch relancera un
# worker neuf plutot que de continuer sur un GPU corrompu.
EXIT_CUDA = 75

# Region explicite plutot qu'implicite : sans elle, boto3 leve NoRegionError
# des l'import, ce qui rend le module intestable hors d'un contexte AWS complet
# et transforme une config manquante en trace de pile illisible.
REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "eu-west-3"
sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
sns = boto3.client("sns", region_name=REGION) if SNS_TOPIC_ARN else None


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def _split_s3(uri):
    reste = uri[5:]
    seau, _, cle = reste.partition("/")
    return seau, cle


class Heartbeat:
    """Prolonge la visibilité du message tant qu'on le traite.

    Sans ça, une vidéo plus longue que le visibility timeout serait redistribuée
    et traitée deux fois. Et si l'instance Spot disparaît, le heartbeat s'arrête
    avec elle : le message redevient visible en moins d'une minute au lieu
    d'attendre les 20 minutes du timeout.
    """

    def __init__(self, receipt):
        self.receipt, self.stop = receipt, threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *_):
        self.stop.set()

    def _run(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            try:
                sqs.change_message_visibility(
                    QueueUrl=QUEUE_URL, ReceiptHandle=self.receipt,
                    VisibilityTimeout=VISIBILITY_SECONDS)
            except Exception as err:                       # noqa: BLE001
                log(f"  heartbeat impossible ({err}) — on continue quand même")
                return


def _preparer_video(job, dossier):
    """Télécharge la vidéo et applique le découpage éventuel."""
    seau, cle = _split_s3(job["s3_input"])
    local = dossier / Path(cle).name
    s3.download_file(seau, cle, str(local))

    debut, fin = job.get("trim_start"), job.get("trim_end")
    if not debut and not fin:
        return local
    coupe = dossier / f"trimmed_{local.name}"
    cmd = ["ffmpeg", "-y", "-ss", str(debut or 0), "-i", str(local)]
    if fin:
        cmd += ["-t", str(int(fin) - int(debut or 0))]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", str(coupe)]
    subprocess.run(cmd, check=True, capture_output=True)
    return coupe


def _compter_images(video):
    """(fps, nombre d'images) par ffprobe ; (None, None) si illisible."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=avg_frame_rate,nb_frames,duration", "-of", "json", str(video)],
            capture_output=True, text=True, timeout=60, check=True)
        st = json.loads(r.stdout)["streams"][0]
        num, den = (st.get("avg_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        n = int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else \
            int(float(st.get("duration") or 0) * fps)
        return (fps or None), (n or None)
    except Exception as err:                                # noqa: BLE001
        log(f"  ffprobe illisible ({err}) — pas de budget d'images applique")
        return None, None


def _budget_images(video, extra_args):
    """Arguments a ajouter pour tenir dans MAX_FRAMES, et la note a publier.

    Renvoie (args, note) ; note vaut None si la video tient dans le budget.
    Respecte un --max_frames / --target_fps deja fourni par le message.
    """
    if MAX_FRAMES <= 0 or "--max_frames" in extra_args:
        return [], None
    fps, n = _compter_images(video)
    if not fps or not n or n <= MAX_FRAMES:
        return [], None
    args, fps_eff = [], fps
    if fps > FPS_REPLI + 5 and "--target_fps" not in extra_args:
        args += ["--target_fps", str(FPS_REPLI)]
        fps_eff = fps / max(1, round(fps / FPS_REPLI))
    n_eff = int(n * fps_eff / fps)
    note = {"frames_total": n, "fps_source": round(fps, 2),
            "fps_processed": round(fps_eff, 2), "max_frames": MAX_FRAMES,
            "truncated": False}
    if n_eff > MAX_FRAMES:
        args += ["--max_frames", str(MAX_FRAMES)]
        note["truncated"] = True
        note["seconds_processed"] = round(MAX_FRAMES / fps_eff, 1)
        note["seconds_total"] = round(n / fps, 1)
    note["message_fr"] = (
        f"Video longue ({n} images, {n / fps:.0f} s) : "
        + (f"seules les {note['seconds_processed']:.0f} premieres secondes "
           f"ont ete analysees" if note["truncated"] else
           f"analysee a {fps_eff:.0f} images/s au lieu de {fps:.0f}")
        + " pour tenir dans la memoire du serveur.")
    log(f"  budget d'images : {note['message_fr']}")
    return args, note


def _envoyer_resultats(dossier_sortie, s3_output):
    seau, prefixe = _split_s3(s3_output)
    prefixe = prefixe.rstrip("/")
    n = 0
    for f in sorted(Path(dossier_sortie).rglob("*")):
        if f.is_file():
            rel = f.relative_to(dossier_sortie).as_posix()
            s3.upload_file(str(f), seau, f"{prefixe}/{rel}")
            n += 1
    return n


def _duree(s):
    return f"{int(s) // 60} min {int(s) % 60:02d} s" if s >= 60 else f"{s:.0f} s"


def _prevenir(job, dossier_sortie, duree, faits=0, erreur=None):
    """Mail de fin. Aligné sur ce que produisait lambda_completion.py.

    Une notification qui ne dit que la durée oblige à aller fouiller S3 pour
    savoir si le résultat est exploitable. On remonte donc ce qui permet de
    juger sans ouvrir quoi que ce soit : volume traité, succès de l'IK,
    qualité de l'ajustement, et de quoi rapatrier les fichiers.
    """
    if sns is None:
        return
    nom = Path(job["s3_input"]).name
    l = [f"Analyse vidéo SAM3D — {'ÉCHEC ⚠' if erreur else 'terminée ✅'}", "",
         f"Fichier       : {nom}",
         f"Traitée par   : worker persistant (vidéo n°{faits} de cette session)"]
    if erreur:
        l += ["", f"Erreur        : {erreur}", "",
              "Le message reste en file : SQS le représentera. Après trois",
              "tentatives il partira en file d'échec (synkro-shared-video-sam3d-dlq)."]
    else:
        if job.get("_note"):
            l += ["", f"⚠ {job['_note']}"]
        l += ["", "─── Temps ───",
              f"Traitement    : {_duree(duree)}",
              "  (le chargement des modèles, ~130 s, n'est PAS repayé :",
              "   c'est tout l'intérêt du worker persistant)",
              f"Coût estimé   : ~{duree / 3600 * 0.21:.4f} $  (g4dn.xlarge Spot eu-west-3)"]

        rapport = Path(dossier_sortie) / "processing_report.json"
        if rapport.exists():
            try:
                r = json.loads(rapport.read_text())
                v, p = r.get("video_info", {}), r.get("processing", {})
                l += ["", "─── Traitement ───"]
                if v.get("width"):
                    l.append(f"Vidéo         : {v.get('width')}x{v.get('height')} "
                             f"@ {v.get('fps', '?')} fps")
                if p.get("num_frames"):
                    l.append(f"Images        : {p['num_frames']}")
                if p.get("num_markers"):
                    l.append(f"Marqueurs     : {p['num_markers']}")
                if "ik_success" in p:
                    l.append(f"OpenSim IK    : {'✓ succès' if p['ik_success'] else '✗ échec'}")
            except Exception as err:                        # noqa: BLE001
                l.append(f"(rapport illisible : {err})")

        # L'erreur d'ajustement dit si les chiffres sont exploitables. Une RMS
        # élevée invalide tout le reste, autant le voir dans le mail.
        for sto in Path(dossier_sortie).rglob("*_ik_marker_errors.sto"):
            try:
                vals = []
                lignes = sto.read_text(errors="replace").splitlines()
                i = next(k for k, x in enumerate(lignes)
                         if x.strip().lower() == "endheader")
                cols = lignes[i + 1].split("\t")
                j = cols.index("marker_error_RMS")
                for x in lignes[i + 2:]:
                    m = x.split()
                    if len(m) > j:
                        vals.append(float(m[j]))
                if vals:
                    l.append(f"Ajustement IK : RMS moyen {sum(vals) / len(vals) * 1000:.1f} mm "
                             f"(max {max(vals) * 1000:.1f} mm)")
            except Exception:                               # noqa: BLE001
                pass
            break

        dossier = job["s3_output"].rstrip("/")
        l += ["", "─── Résultats ───", f"Dossier       : {dossier}/", "",
              "Rapatrier en local :",
              f"  aws s3 sync {dossier}/ ./{Path(nom).stem}/ --region {REGION}"]

    titre = f"[SAM3D] {'⚠ Échec' if erreur else '✅ Terminée'} — {nom}"
    try:
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=titre[:100],
                    Message="\n".join(l))
    except Exception as err:                                # noqa: BLE001
        log(f"  SNS indisponible ({err}) — le job reste valide")


def traiter(session, job, dossier):
    """Traite une vidéo. Renvoie la durée. Lève si le job échoue."""
    video = _preparer_video(job, dossier)
    sortie = dossier / "out"
    argv = [
        "--video_path", str(video),
        "--output_dir", str(sortie),
        "--inference_type", job.get("inference_type", "body"),
        "--markerset", "flodelaplace",
        "--floor_moge",
        "--detector_model", "checkpoints/yolo/yolo11m-pose.engine",
        "--bbox_thr", "0.2", "--nms_thr", "0.9",
        "--detect_then_infer", "--inference_batch_cap", "4",
        "--fallback_lower_bbox", "0.05", "--fallback_nms", "0.9",
        "--fallback_iou_thresh", "0.5",
    ] + shlex.split(job.get("extra_args", ""))
    ajout, note = _budget_images(video, job.get("extra_args", ""))
    argv += ajout

    t = time.time()
    session.run(argv)
    duree = time.time() - t
    if note:
        job["_note"] = note["message_fr"]
        sortie.mkdir(parents=True, exist_ok=True)
        (sortie / "analysis_note.json").write_text(
            json.dumps(note, ensure_ascii=False, indent=2))
    n = _envoyer_resultats(sortie, job["s3_output"])
    log(f"  {n} fichiers envoyés vers {job['s3_output']}")
    return duree


def main():
    import tempfile

    from worker.session import InferenceSession

    depart = time.time()
    log(f"démarrage — file {QUEUE_URL.rsplit('/', 1)[-1]}, "
        f"extinction après {IDLE_EXIT_SECONDS} s d'inactivité")
    session = InferenceSession()
    log(f"modèles chargés en {time.time() - depart:.1f} s")

    dernier = time.time()
    faits = 0
    while True:
        if faits >= MAX_JOBS:
            log(f"{faits} vidéos traitées — redémarrage préventif"); return 0
        if time.time() - depart > MAX_AGE_SECONDS:
            log("âge maximum atteint — redémarrage préventif"); return 0
        if time.time() - dernier > IDLE_EXIT_SECONDS:
            log(f"aucun message depuis {IDLE_EXIT_SECONDS} s — extinction"); return 0

        rep = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=1,
                                  WaitTimeSeconds=20,
                                  VisibilityTimeout=VISIBILITY_SECONDS)
        messages = rep.get("Messages", [])
        if not messages:
            a, r = session.gpu_memory()
            log(f"en attente — {faits} traitées, GPU {a:.0f}/{r:.0f} Mo, "
                f"inactif depuis {time.time() - dernier:.0f} s")
            continue

        msg = messages[0]
        recu = msg["ReceiptHandle"]
        try:
            job = json.loads(msg["Body"])
        except json.JSONDecodeError as err:
            log(f"message illisible ({err}) — supprimé, il partirait en boucle")
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=recu)
            continue

        dernier = time.time()
        log(f"=== {job.get('s3_input', '?')}")
        with tempfile.TemporaryDirectory(prefix="synkro-") as tmp, Heartbeat(recu):
            try:
                duree = traiter(session, job, Path(tmp))
                sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=recu)
                faits += 1
                a, r = session.gpu_memory()
                log(f"  terminé en {duree:.0f} s — GPU {a:.0f}/{r:.0f} Mo")
                _prevenir(job, Path(tmp) / "out", duree, faits=faits)
            except KeyboardInterrupt:
                raise
            except BaseException as err:                    # noqa: BLE001
                # Un etat CUDA casse ne se repare pas : continuer produirait des
                # resultats faux et silencieux. On sort, Batch relance propre.
                texte = f"{type(err).__name__}: {err}"
                if "CUDA" in texte or "cuda" in type(err).__name__.lower():
                    log(f"  ERREUR CUDA — {texte}")
                    log("  le GPU est dans un état irrécupérable, on redémarre")
                    return EXIT_CUDA
                # Erreur metier : on NE supprime PAS le message. SQS le rendra,
                # et apres 3 tentatives il partira en file d'echec.
                log(f"  ÉCHEC — {texte}")
                _prevenir(job, Path(tmp) / "out", time.time() - dernier,
                          faits=faits + 1, erreur=texte)
        # L'inactivite se compte a partir de la FIN du traitement, pas de la
        # reception. Vu le 2026-09-21 : une video de 659 s, puis « aucun
        # message depuis 600 s — extinction » sans un seul receive_message,
        # alors qu'un message deja depose attendait dans la file. Rien ne
        # reveille un worker sur un message existant (la lambda ne reagit
        # qu'aux depots) : la video restait bloquee jusqu'au depot suivant.
        dernier = time.time()


if __name__ == "__main__":
    sys.path.insert(0, "/app")
    sys.exit(main())
