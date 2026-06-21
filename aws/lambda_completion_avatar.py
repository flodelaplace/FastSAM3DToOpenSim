"""
Lambda triggered by EventBridge on AWS Batch job state changes (SUCCEEDED/FAILED)
for the synkro-fastsam3d-avatar-job (queue: synkro-fastsam3d-queue, filtered by
job name prefix 'avatar-').

Publishes a SNS notification with:
  - timing breakdown (queue_wait + boot_pull + container, same logic as SAM3D)
  - cost estimate (g4dn.xlarge Spot)
  - list of generated avatar GLBs in S3
  - CloudWatch log stream pointer
"""
from __future__ import annotations

import json
import os

import boto3

SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
S3_OUTPUT_BUCKET = os.environ.get("S3_OUTPUT_BUCKET", "data-synchro-video")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "02-output-avatar")

# g4dn.xlarge Spot eu-west-3 — approximate.
SPOT_PRICE_USD_PER_HOUR = 0.16
WARM_RESTART_MS = 30_000


def _fmt_duration(ms):
    if ms is None or ms < 0:
        return "—"
    s = ms / 1000.0
    m, sec = divmod(s, 60)
    return f"{int(m)} min {int(sec):02d} s" if m >= 1 else f"{sec:.1f} s"


def _get_env(detail, name):
    env = detail.get("container", {}).get("environment", []) or []
    return next((e["value"] for e in env if e.get("name") == name), None)


def _list_avatar_glbs(s3_output_uri):
    """List the .glb files generated under <s3_output_uri>/."""
    if not s3_output_uri or not s3_output_uri.startswith("s3://"):
        return []
    # Strip s3://bucket/key/ → bucket, prefix
    no_scheme = s3_output_uri[len("s3://"):]
    bucket, _, prefix = no_scheme.partition("/")
    s3 = boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [
            f"s3://{bucket}/{obj['Key']}"
            for obj in resp.get("Contents", [])
            if obj["Key"].endswith(".glb")
        ]
    except Exception as err:
        print(f"WARN: list_objects failed: {err}")
        return []


def _estimate_cost(container_ms):
    if container_ms is None:
        return None
    hours = (container_ms / 1000.0 + 120) / 3600.0
    return hours * SPOT_PRICE_USD_PER_HOUR


def _get_instance_launch_ms(job_id):
    """EC2 launch_time via Batch→ECS→EC2. None on failure (then no split)."""
    try:
        batch = boto3.client("batch")
        jobs = batch.describe_jobs(jobs=[job_id]).get("jobs", [])
        if not jobs:
            return None
        attempts = jobs[0].get("attempts", [])
        if not attempts:
            return None
        last = attempts[-1]
        ci_arn = last.get("container", {}).get("containerInstanceArn")
        if not ci_arn:
            return None
        parts = ci_arn.split("/")
        if len(parts) < 3:
            return None
        cluster = parts[1]
        ecs = boto3.client("ecs")
        resp = ecs.describe_container_instances(
            cluster=cluster, containerInstances=[ci_arn]
        )
        instances = resp.get("containerInstances", [])
        if not instances:
            return None
        ec2_id = instances[0].get("ec2InstanceId")
        if not ec2_id:
            return None
        ec2 = boto3.client("ec2")
        r = ec2.describe_instances(InstanceIds=[ec2_id])
        for rr in r.get("Reservations", []):
            for inst in rr.get("Instances", []):
                lt = inst.get("LaunchTime")
                if lt is not None:
                    return int(lt.timestamp() * 1000)
    except Exception as err:
        print(f"WARN: _get_instance_launch_ms failed: {err}")
    return None


def _split_pre_container_time(created_at, started_at, launch_at):
    if not (created_at and started_at):
        return None, None, "unknown"
    if launch_at is None:
        return None, None, "unknown"
    if launch_at > created_at:
        queue_wait = launch_at - created_at
        boot_pull = max(0, started_at - launch_at)
        regime = "fresh"
    else:
        total = started_at - created_at
        boot_pull = min(WARM_RESTART_MS, total)
        queue_wait = max(0, total - boot_pull)
        regime = "warm"
    return queue_wait, boot_pull, regime


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

    s3_output_uri = _get_env(detail, "S3_OUTPUT_URI") or ""
    s3_input_uri = _get_env(detail, "S3_INPUT_URI") or ""

    # job name format: avatar-<user>-<exercise>
    raw_name = job_name.removeprefix("avatar-")

    cold_start_ms = (started_at - created_at) if (started_at and created_at) else None
    container_ms = (stopped_at - started_at) if (stopped_at and started_at) else None
    total_ms = (stopped_at - created_at) if (stopped_at and created_at) else None
    cost_usd = _estimate_cost(container_ms)

    launch_at = _get_instance_launch_ms(job_id)
    queue_wait_ms, boot_pull_ms, regime = _split_pre_container_time(
        created_at, started_at, launch_at
    )

    glbs = _list_avatar_glbs(s3_output_uri) if status == "SUCCEEDED" else []

    emoji = "✅" if status == "SUCCEEDED" else "❌"
    subject = f"[AVATAR] {emoji} {status} — {raw_name}"

    lines = [f"Génération avatars SAM3D — {status} {emoji}", ""]
    lines.append(f"Job name      : {job_name}")
    lines.append(f"Job ID        : {job_id}")
    if exit_code is not None:
        lines.append(f"Exit code     : {exit_code}")
    if status != "SUCCEEDED" and status_reason:
        lines.append(f"Raison échec  : {status_reason}")
    lines.append("")

    lines.append("─── Temps d'exécution ───")
    if queue_wait_ms is not None and boot_pull_ms is not None:
        q_lbl = "attente lancement instance" if regime == "fresh" else "attente fin job précédent"
        b_lbl = "boot Spot + pull image ECR" if regime == "fresh" else "warm restart (instance déjà chaude)"
        lines.append(f"Queue wait    : {_fmt_duration(queue_wait_ms)}  ({q_lbl})")
        lines.append(f"Boot + pull   : {_fmt_duration(boot_pull_ms)}  ({b_lbl})")
    else:
        lines.append(f"Cold start    : {_fmt_duration(cold_start_ms)}")
    lines.append(f"Container     : {_fmt_duration(container_ms)}  (sync + inference + retarget + upload)")
    lines.append(f"Total         : {_fmt_duration(total_ms)}")
    lines.append("")

    if status == "SUCCEEDED":
        lines.append("─── Avatars générés ───")
        if glbs:
            for g in sorted(glbs):
                lines.append(f"  {g}")
        else:
            lines.append("  (aucun .glb détecté à l'emplacement de sortie)")
        lines.append("")

    lines.append("─── Coût ───")
    if cost_usd is not None:
        lines.append(f"Estimé        : ~{cost_usd:.4f} $  (g4dn.xlarge Spot eu-west-3)")
    lines.append("")

    if s3_input_uri:
        lines.append(f"Source        : {s3_input_uri}")
    if s3_output_uri:
        lines.append(f"Output dir    : {s3_output_uri}")

    if log_stream:
        lines.append("")
        lines.append(f"Log stream    : /aws/batch/job/{log_stream}")

    sns = boto3.client("sns")
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject[:100],
        Message="\n".join(lines),
    )
    return {"statusCode": 200, "body": "ok"}
