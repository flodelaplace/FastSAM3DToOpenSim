"""
Lambda triggered by EventBridge on AWS Batch job state changes (SUCCEEDED/FAILED)
for the synkro-fastsam3d-queue.

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
    "S3_OUTPUT_URI", "s3://data-synchro-video/02-output-SAM3D/"
)

# g4dn.xlarge Spot eu-west-3 — approximate (price varies ±10%).
SPOT_PRICE_USD_PER_HOUR = 0.16


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

    raw_name = job_name.removeprefix("fastsam-")
    video_name = _get_video_name(detail) or raw_name

    cold_start_ms = (started_at - created_at) if (started_at and created_at) else None
    container_ms = (stopped_at - started_at) if (stopped_at and started_at) else None
    total_ms = (stopped_at - created_at) if (stopped_at and created_at) else None
    cost_usd = _estimate_cost(container_ms)

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
        lines.append(f"Group  : /aws/batch/synkro-fastsam3d")
        lines.append(f"Stream : {log_stream}")
        lines.append("")
        lines.append(
            "Voir :  aws logs tail /aws/batch/synkro-fastsam3d "
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
