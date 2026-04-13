"""
Lambda function that triggers an AWS Batch job when a video lands in S3.

Triggered by S3 event on s3://data-synchro-video/01-input/*.

Submits a Batch job to the synkro-fastsam3d-queue with the video S3 URI
as S3_INPUT_URI override.
"""
import json
import os
import re
import urllib.parse

import boto3

batch = boto3.client("batch")

JOB_QUEUE = os.environ["JOB_QUEUE"]            # synkro-fastsam3d-queue
JOB_DEFINITION = os.environ["JOB_DEFINITION"]  # synkro-fastsam3d-job

# Accepted video extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Sanitize job name: only alphanumeric, hyphen, underscore (max 128 chars)
_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _job_name_from_key(key: str) -> str:
    base = os.path.basename(key)
    name = os.path.splitext(base)[0]
    safe = _SANITIZE_RE.sub("-", name)[:100]
    return f"fastsam-{safe}"


def lambda_handler(event, context):
    submitted = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        ext = os.path.splitext(key)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            print(f"Skipping non-video file: {key}")
            continue

        s3_input = f"s3://{bucket}/{key}"
        job_name = _job_name_from_key(key)

        print(f"Submitting Batch job for {s3_input}")

        response = batch.submit_job(
            jobName=job_name,
            jobQueue=JOB_QUEUE,
            jobDefinition=JOB_DEFINITION,
            containerOverrides={
                "environment": [
                    {"name": "S3_INPUT_URI", "value": s3_input},
                ]
            },
        )

        submitted.append({
            "jobId": response["jobId"],
            "jobName": response["jobName"],
            "input": s3_input,
        })

    return {
        "statusCode": 200,
        "body": json.dumps({"submitted": submitted}),
    }
