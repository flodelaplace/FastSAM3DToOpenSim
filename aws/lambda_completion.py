"""
Lambda triggered by EventBridge on AWS Batch job state changes (SUCCEEDED/FAILED)
for the synkro-shared-video-sam3d-queue.

Publishes a detailed notification via SNS including:
  - timing breakdown (cold start vs container time)
  - inference stats (frames, markers, IK success) from processing_report.json
  - cost estimate (g4dn.xlarge Spot)
  - S3 output folder URL + download command
  - CloudWatch log stream pointer
"""
from __future__ import annotations

import json
import os

import boto3

SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
S3_OUTPUT_URI = os.environ.get(
    "S3_OUTPUT_URI", "s3://synkro-shared-video/02-output-SAM3D/"
)

# g4dn.xlarge Spot eu-west-3 — approximate (price varies ±10%).
SPOT_PRICE_USD_PER_HOUR = 0.16
# Prefixe des jobs worker persistant, aligne sur aws/lambda_trigger.py.
WORKER_NAME_PREFIX = os.environ.get("WORKER_NAME_PREFIX", "fastsam-worker")
WORK_QUEUE_URL = os.environ.get(
    "WORK_QUEUE_URL",
    "https://sqs.eu-west-3.amazonaws.com/763695379040/synkro-shared-video-sam3d-work")
JOB_QUEUE = os.environ.get("JOB_QUEUE", "synkro-shared-video-sam3d-queue")
WORKER_JOB_DEFINITION = os.environ.get("WORKER_JOB_DEFINITION",
                                       "synkro-shared-video-sam3d-worker-job")


def _relancer_worker_si_file():
    """Relance un worker si la file SQS n'est pas vide et qu'aucun ne tourne.

    Rien d'autre ne le fait : la lambda de depot ne reveille un worker qu'a
    l'arrivee d'une NOUVELLE video. Vu le 2026-09-24 (cours de M2, 29 videos) :
    le worker meurt (memoire), 11 videos restent en file, aucun job ne tourne
    et rien ne bouge jusqu'au prochain depot. Meme trou quand un worker sort
    NORMALEMENT (20 videos traitees, ou age maximum) avec une file encore
    pleine. Renvoie une phrase pour le mail.
    """
    try:
        sqs = boto3.client("sqs")
        a = sqs.get_queue_attributes(
            QueueUrl=WORK_QUEUE_URL,
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
        en_file = int(a.get("ApproximateNumberOfMessages", 0))
        en_cours = int(a.get("ApproximateNumberOfMessagesNotVisible", 0))
        if en_file + en_cours == 0:
            return "File vide : aucun worker relance."
        batch = boto3.client("batch")
        for statut in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
            for j in batch.list_jobs(jobQueue=JOB_QUEUE, jobStatus=statut,
                                     maxResults=100).get("jobSummaryList", []):
                if j.get("jobName", "").startswith(WORKER_NAME_PREFIX):
                    return (f"{en_file + en_cours} video(s) en file ; un autre "
                            f"worker tourne deja ({statut}).")
        r = batch.submit_job(jobName=WORKER_NAME_PREFIX, jobQueue=JOB_QUEUE,
                             jobDefinition=WORKER_JOB_DEFINITION)
        print(f"WORKER RELANCE {r['jobId']} ({en_file}+{en_cours} en file)")
        return (f"{en_file + en_cours} video(s) en file : NOUVEAU worker lance "
                f"automatiquement (job {r['jobId']}).")
    except Exception as err:                                  # noqa: BLE001
        print(f"WARN relance worker impossible : {err}")
        return f"ATTENTION : relance automatique impossible ({err}). A relancer a la main."


def _prevenir_worker_eteint(job_id, job_name, created_at, started_at,
                            stopped_at, log_stream):
    """Le worker s'est eteint normalement — le dire, et dire que ce n'en est pas une.

    Sans ce mail, l'extinction serait silencieuse. Avec l'ancien mail generique,
    elle passait pour « une analyse terminee en 14 min 39 » — alors que ces 14
    minutes couvrent DEUX videos traitees en 131 et 114 s, plus dix minutes
    d'attente avant extinction. Le titre et la premiere ligne doivent donc lever
    l'ambiguite avant meme qu'on lise les chiffres.
    """
    demarrage = (started_at - created_at) if (started_at and created_at) else None
    session = (stopped_at - started_at) if (stopped_at and started_at) else None
    cout = (session / 3_600_000 * SPOT_PRICE_USD_PER_HOUR) if session else None

    relance = _relancer_worker_si_file()
    l = ["EXTINCTION DU WORKER — ceci n'est PAS une analyse.", "",
         relance, "",
         "Le worker persistant s'est eteint apres son delai d'inactivite, et la",
         "machine GPU a ete liberee. C'est le fonctionnement normal : il se",
         "rallumera au prochain depot de video.", "",
         "Les analyses qu'il a traitees ont fait l'objet d'un mail CHACUNE.",
         "Les durees ci-dessous sont celles de la SESSION, pas d'une analyse.", "",
         f"Job           : {job_name}", f"Job ID        : {job_id}"]
    if demarrage is not None:
        l.append(f"Demarrage     : {_fmt_duration(demarrage)}  (boot Spot + image)")
    if session is not None:
        l.append(f"Session       : {_fmt_duration(session)}  (chargement des modeles"
                 " + analyses + attente)")
    if cout is not None:
        l.append(f"Cout session  : ~{cout:.4f} $  (g4dn.xlarge Spot eu-west-3)")
    if log_stream:
        l += ["", "─── Logs ───",
              f"aws logs tail /aws/batch/synkro-shared-video-sam3d "
              f"--log-stream-names {log_stream} --region eu-west-3"]
    boto3.client("sns").publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="[SAM3D] 💤 Worker eteint (fin de session, pas une analyse)"[:100],
        Message="\n".join(l))
    print(f"WORKER SHUTDOWN {job_name} — mail d'extinction envoye")
    return {"statusCode": 200, "body": "worker shutdown notified"}


def _prevenir_worker_en_echec(detail, job_id, job_name, status, status_reason,
                              exit_code, log_stream):
    """Alerte quand un worker meurt anormalement.

    Un worker qui s'eteint apres son delai d'inactivite est un evenement normal
    et silencieux. Un worker qui tombe, en revanche, laisse des videos en file
    sans personne pour les traiter — et c'est le seul cas ou il faut prevenir.

    Le code 75 est reserve aux etats CUDA irrecuperables (voir worker/loop.py) :
    le worker sort volontairement pour que Batch en relance un propre, plutot
    que de continuer sur un GPU corrompu et produire des resultats faux.
    """
    lignes = [f"Worker persistant SAM3D — {status} ⚠", "",
              f"Job           : {job_name}", f"Job ID        : {job_id}"]
    if exit_code is not None:
        lignes.append(f"Exit code     : {exit_code}")
        if exit_code == 75:
            lignes += ["",
                       "Code 75 = erreur CUDA. Le worker s'est arrete VOLONTAIREMENT :",
                       "l'etat du GPU etait irrecuperable et continuer aurait produit des",
                       "resultats faux sans le signaler. Un worker propre est relance",
                       "ci-dessous si des videos attendent.",
                       "Les videos en cours reviennent en file automatiquement."]
    if status_reason:
        lignes.append(f"Raison        : {status_reason}")
    relance = _relancer_worker_si_file()
    lignes += ["", relance, "",
               "Les videos non traitees restent dans la file SQS et sont reprises",
               "par le worker suivant. Une video qui fait tomber le worker trois",
               "fois part dans synkro-shared-video-sam3d-dlq.", ""]
    if log_stream:
        lignes += ["─── Logs ───",
                   f"aws logs tail /aws/batch/synkro-shared-video-sam3d "
                   f"--log-stream-names {log_stream} --region eu-west-3"]
    boto3.client("sns").publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"[SAM3D] ⚠ Worker en echec — {status}"[:100],
        Message="\n".join(lignes))
    return {"statusCode": 200, "body": "worker failure notified"}

# Estimated time for a warm container restart on an already-running instance
# (docker container restart + TRT engine cache fetch from S3). Used to split
# queue_wait from boot_pull when the instance was already up before the job.
WARM_RESTART_MS = 30_000


def _fmt_duration(ms):
    if ms is None or ms < 0:
        return "—"
    s = ms / 1000.0
    m, sec = divmod(s, 60)
    return f"{int(m)} min {int(sec):02d} s" if m >= 1 else f"{sec:.1f} s"


def _parse_s3_uri(uri):
    stripped = uri.replace("s3://", "").split("/", 1)
    bucket = stripped[0]
    key_prefix = stripped[1] if len(stripped) > 1 else ""
    return bucket, key_prefix


def _get_video_name(detail):
    """Extract video basename (without extension) from Batch job env overrides."""
    env = detail.get("container", {}).get("environment", [])
    s3_input = next((e["value"] for e in env if e.get("name") == "S3_INPUT_URI"), None)
    if not s3_input:
        return None
    return os.path.splitext(os.path.basename(s3_input))[0]


def _find_output_folder(video_name):
    """Return s3://bucket/prefix/output_*_<video_name>/ or None."""
    bucket, prefix = _parse_s3_uri(S3_OUTPUT_URI)
    s3 = boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        matches = [
            p["Prefix"]
            for p in resp.get("CommonPrefixes", [])
            if p["Prefix"].rstrip("/").endswith(f"_{video_name}")
        ]
        if not matches:
            return None
        return f"s3://{bucket}/{sorted(matches)[-1]}"
    except Exception as err:
        print(f"WARN: list_objects failed: {err}")
        return None


def _fetch_report(output_folder):
    """Download processing_report.json if it exists."""
    if not output_folder:
        return {}
    bucket, key = _parse_s3_uri(output_folder.rstrip("/") + "/processing_report.json")
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as err:
        print(f"WARN: could not fetch processing_report.json: {err}")
        return {}


def _estimate_cost(container_ms):
    """Estimate cost USD — bill covers container run + ~2 min boot overhead."""
    if container_ms is None:
        return None
    hours = (container_ms / 1000.0 + 120) / 3600.0
    return hours * SPOT_PRICE_USD_PER_HOUR


def _get_instance_launch_ms(job_id):
    """Return the EC2 launch time (ms epoch) of the instance that ran this job.

    Walks: Batch DescribeJobs → containerInstanceArn → ECS DescribeContainerInstances
           → ec2InstanceId → EC2 DescribeInstances → LaunchTime.

    Returns None on any failure (so the caller falls back to legacy single-bucket).
    """
    try:
        batch = boto3.client("batch")
        jobs = batch.describe_jobs(jobs=[job_id]).get("jobs", [])
        if not jobs:
            return None
        attempts = jobs[0].get("attempts", [])
        if not attempts:
            return None
        # Last attempt is the one that ran to SUCCEEDED/FAILED
        last = attempts[-1]
        container_instance_arn = last.get("container", {}).get("containerInstanceArn")
        if not container_instance_arn:
            return None
        # The container_instance_arn looks like
        # arn:aws:ecs:<region>:<acct>:container-instance/<cluster>/<id>
        # We need the cluster part for DescribeContainerInstances.
        # Format: container-instance/<cluster>/<uuid>
        parts = container_instance_arn.split("/")
        if len(parts) < 3:
            return None
        cluster = parts[1]
        ecs = boto3.client("ecs")
        resp = ecs.describe_container_instances(
            cluster=cluster, containerInstances=[container_instance_arn]
        )
        instances = resp.get("containerInstances", [])
        if not instances:
            return None
        ec2_id = instances[0].get("ec2InstanceId")
        if not ec2_id:
            return None
        ec2 = boto3.client("ec2")
        resp = ec2.describe_instances(InstanceIds=[ec2_id])
        for r in resp.get("Reservations", []):
            for inst in r.get("Instances", []):
                lt = inst.get("LaunchTime")
                if lt is not None:
                    # Convert datetime → ms epoch
                    return int(lt.timestamp() * 1000)
    except Exception as err:
        print(f"WARN: _get_instance_launch_ms failed: {err}")
    return None


def _split_pre_container_time(created_at, started_at, launch_at):
    """Decompose started_at - created_at into queue_wait_ms + boot_pull_ms.

    Two regimes:
      - launch_at > created_at → instance was created (or in process of launching)
        AFTER job submission. The job waited (launch_at - created_at) for the
        instance to spawn, then (started_at - launch_at) for boot + image pull.
      - launch_at <= created_at → instance was already running when the job was
        submitted. The job waited for the previous job/capacity to free up. A
        small fixed amount is attributed to container restart (warm path).
    """
    if not (created_at and started_at):
        return None, None, "unknown"
    if launch_at is None:
        return None, None, "unknown"
    if launch_at > created_at:
        queue_wait_ms = launch_at - created_at
        boot_pull_ms = max(0, started_at - launch_at)
        regime = "fresh"
    else:
        # Warm path: instance was already up
        total = started_at - created_at
        boot_pull_ms = min(WARM_RESTART_MS, total)
        queue_wait_ms = max(0, total - boot_pull_ms)
        regime = "warm"
    return queue_wait_ms, boot_pull_ms, regime


def lambda_handler(event, context):
    detail = event.get("detail", {})
    job_id = detail.get("jobId", "?")
    job_name = detail.get("jobName", "?")
    status = detail.get("status", "UNKNOWN")
    status_reason = detail.get("statusReason", "")
    created_at = detail.get("createdAt")
    started_at = detail.get("startedAt")
    stopped_at = detail.get("stoppedAt")
    container = detail.get("container", {}) or {}
    exit_code = container.get("exitCode")
    log_stream = container.get("logStreamName", "")

    # Un worker persistant n'est pas une analyse : il en enchaine plusieurs puis
    # attend son delai d'inactivite avant de s'eteindre. Ce mail-ci, declenche
    # par la fin du JOB Batch, annoncerait donc « analyse terminee en 14 min 39 »
    # pour deux videos traitees en 131 s et 114 s, suivies de dix minutes
    # d'attente. Exactement la mauvaise impression : celle d'une analyse qui rame.
    #
    # C'est le worker lui-meme qui notifie, video par video (worker/loop.py).
    # Ici on se contente d'un recapitulatif de session, et seulement s'il a
    # echoue — un worker qui s'eteint normalement n'interesse personne.
    if job_name.startswith(WORKER_NAME_PREFIX):
        if status == "SUCCEEDED":
            return _prevenir_worker_eteint(job_id, job_name, created_at,
                                           started_at, stopped_at, log_stream)
        print(f"WORKER FAILED {job_name} — mail d'alerte")
        return _prevenir_worker_en_echec(detail, job_id, job_name, status,
                                         status_reason, exit_code, log_stream)

    raw_name = job_name.removeprefix("fastsam-")
    video_name = _get_video_name(detail) or raw_name

    cold_start_ms = (started_at - created_at) if (started_at and created_at) else None
    container_ms = (stopped_at - started_at) if (stopped_at and started_at) else None
    total_ms = (stopped_at - created_at) if (stopped_at and created_at) else None
    cost_usd = _estimate_cost(container_ms)

    # Split cold_start_ms into queue_wait + boot_pull using EC2 instance launch time
    launch_at = _get_instance_launch_ms(job_id)
    queue_wait_ms, boot_pull_ms, regime = _split_pre_container_time(
        created_at, started_at, launch_at
    )

    output_folder = _find_output_folder(video_name) if status == "SUCCEEDED" else None
    report = _fetch_report(output_folder) if output_folder else {}

    emoji = "✅" if status == "SUCCEEDED" else "❌"
    subject = f"[SAM3D] {emoji} {status} — {raw_name}"

    lines = [f"Analyse vidéo SAM3D — {status} {emoji}", ""]
    lines.append(f"Fichier       : {video_name}")
    lines.append(f"Job ID        : {job_id}")
    if exit_code is not None:
        lines.append(f"Exit code     : {exit_code}")
    if status != "SUCCEEDED" and status_reason:
        lines.append(f"Raison échec  : {status_reason}")
    lines.append("")

    lines.append("─── Temps d'exécution ───")
    if queue_wait_ms is not None and boot_pull_ms is not None:
        boot_label = (
            "boot Spot + pull image ECR" if regime == "fresh"
            else "warm restart (instance déjà chaude)"
        )
        queue_label = (
            "attente lancement instance" if regime == "fresh"
            else "attente fin job précédent"
        )
        lines.append(f"Queue wait    : {_fmt_duration(queue_wait_ms)}  ({queue_label})")
        lines.append(f"Boot + pull   : {_fmt_duration(boot_pull_ms)}  ({boot_label})")
    else:
        lines.append(f"Cold start    : {_fmt_duration(cold_start_ms)}  (boot Spot + pull image ECR)")
    lines.append(f"Container     : {_fmt_duration(container_ms)}  (sync checkpoints + inférence + upload)")
    lines.append(f"Total         : {_fmt_duration(total_ms)}")
    lines.append("")

    if report:
        proc = report.get("processing", {}) or {}
        vid = report.get("video_info", {}) or {}
        timings = report.get("timings", {}) or {}
        lines.append("─── Détails traitement ───")
        if vid:
            lines.append(
                f"Vidéo source  : {vid.get('width', '?')}×{vid.get('height', '?')}"
                f" @ {vid.get('fps', '?')} fps, {vid.get('frame_count', '?')} frames"
            )
        if proc.get("num_frames") is not None:
            lines.append(f"Frames traitées : {proc['num_frames']}")
        if proc.get("num_markers") is not None:
            lines.append(f"Markers extraits : {proc['num_markers']}")
        if timings.get("total") is not None:
            lines.append(f"Pipeline Python : {timings['total']:.1f} s")
        if proc.get("ik_success") is not None:
            lines.append(f"OpenSim IK    : {'✓ success' if proc['ik_success'] else '✗ échec'}")
        lines.append("")

    if cost_usd is not None:
        lines.append("─── Coût ───")
        lines.append(f"Estimé        : ~{cost_usd:.4f} $  (g4dn.xlarge Spot eu-west-3)")
        lines.append("")

    if output_folder:
        lines.append("─── Résultats S3 ───")
        lines.append(f"Dossier       : {output_folder}")
        lines.append("")
        lines.append("Téléchargement local :")
        local_dir = raw_name or "output"
        lines.append(f"  aws s3 sync {output_folder} ./{local_dir}/ --region eu-west-3")
        lines.append("")

    if log_stream:
        lines.append("─── Logs CloudWatch ───")
        lines.append(f"Group  : /aws/batch/synkro-shared-video-sam3d")
        lines.append(f"Stream : {log_stream}")
        lines.append("")
        lines.append(
            "Voir :  aws logs tail /aws/batch/synkro-shared-video-sam3d "
            f"--log-stream-names {log_stream} --region eu-west-3"
        )

    message = "\n".join(lines)

    sns = boto3.client("sns")
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject[:100],
        Message=message,
    )
    return {"statusCode": 200, "body": "notified"}
