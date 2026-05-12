"""
Lambda function that triggers an AWS Batch job when a video lands in S3.

Triggered by S3 event on s3://data-synchro-video/01-input-SAM3D/*.

Filename convention (strict — files not matching are rejected):

    <nom_libre>__<meta_tokens>.<ext>

Tokens:
    h<cm>                required, e.g. h185 -> person_height=1.85m
    h<cm>-<cm>[-...]     multi-person, e.g. h185-170 -> 2 persons 1.85m + 1.70m
    s<sec>               trim start in seconds (integer, default 0)
    e<sec>               trim end   in seconds (integer, default video end)
    st                   --stationary
    com                  --compute_com
    floor                --floor  (mise au sol + redressement caméra : à utiliser
                         pour mouvements debout type squat/marche. Omettre pour
                         rameur, couché, suspension, etc. où le sujet ne doit
                         pas être forcé au sol.)

Examples:
    squat_jean__h185.mp4              single, full video
    cmj__h185_st_com.mp4              single, stationary, compute CoM
    violon__h185_s3_e12.mp4           single, trim 3-12s
    groupe__h185-170-180.mp4          multi-person (auto)
    groupe__h185-170_st_com.mp4       multi, stationary, CoM
"""
import json
import os
import re
import sys
import urllib.parse

try:
    import boto3
except ImportError:
    boto3 = None  # Allow local unit tests without boto3

JOB_QUEUE = os.environ.get("JOB_QUEUE", "synkro-fastsam3d-queue")
JOB_DEFINITION = os.environ.get("JOB_DEFINITION", "synkro-fastsam3d-job")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

HEIGHT_RE = re.compile(r"^h(\d{2,3}(?:-\d{2,3})*)$")
START_RE = re.compile(r"^s(\d+)$")
END_RE = re.compile(r"^e(\d+)$")
FLAG_TOKENS = {"st", "com", "floor"}

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


class FilenameParseError(ValueError):
    pass


def parse_filename(basename):
    """
    Parse a video filename into AWS Batch env-var overrides.

    Returns dict:
        raw_name:    original name without extension (kept as-is for output folder)
        extra_args:  string of flags for demo_video_opensim.py
        trim_start:  str (integer seconds) or ""
        trim_end:    str (integer seconds) or ""
    Raises FilenameParseError on any validation failure.
    """
    name, ext = os.path.splitext(basename)
    ext = ext.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise FilenameParseError(f"Unsupported extension: {ext}")
    if "__" not in name:
        raise FilenameParseError(
            f"Missing '__<meta>' block. "
            f"Expected: <name>__h<cm>[_s<sec>][_e<sec>][_st][_com]{ext}"
        )

    raw_name, _, meta_block = name.rpartition("__")
    if not raw_name:
        raise FilenameParseError("Empty base name before '__'")
    tokens = [t for t in meta_block.split("_") if t]
    if not tokens:
        raise FilenameParseError("Empty meta block after '__'")

    heights = None
    trim_start = None
    trim_end = None
    stationary = False
    compute_com = False
    floor = False

    for t in tokens:
        m = HEIGHT_RE.match(t)
        if m:
            if heights is not None:
                raise FilenameParseError(f"Duplicate height token: '{t}'")
            heights = [int(v) / 100.0 for v in m.group(1).split("-")]
            continue
        m = START_RE.match(t)
        if m:
            if trim_start is not None:
                raise FilenameParseError(f"Duplicate start token: '{t}'")
            trim_start = int(m.group(1))
            continue
        m = END_RE.match(t)
        if m:
            if trim_end is not None:
                raise FilenameParseError(f"Duplicate end token: '{t}'")
            trim_end = int(m.group(1))
            continue
        if t == "st":
            stationary = True
            continue
        if t == "com":
            compute_com = True
            continue
        if t == "floor":
            floor = True
            continue
        raise FilenameParseError(f"Unknown token: '{t}'")

    if heights is None:
        raise FilenameParseError("Missing required token 'h<cm>'")
    if trim_start is not None and trim_end is not None and trim_start >= trim_end:
        raise FilenameParseError(
            f"start ({trim_start}s) must be < end ({trim_end}s)"
        )

    extra = []
    if len(heights) == 1:
        extra += ["--person_height", f"{heights[0]:.2f}"]
    else:
        extra += [
            "--multi_person",
            "--person_heights", ",".join(f"{h:.2f}" for h in heights),
            "--run_ik_per_person",
            "--write_combined_trc",
        ]
    if stationary:
        extra.append("--stationary")
    if compute_com:
        extra.append("--compute_com")
    if floor:
        extra.append("--floor")

    return {
        "raw_name": raw_name,
        "extra_args": " ".join(extra),
        "trim_start": str(trim_start) if trim_start is not None else "",
        "trim_end": str(trim_end) if trim_end is not None else "",
    }


def _job_name(raw_name):
    safe = _SANITIZE_RE.sub("-", raw_name)[:100] or "video"
    return f"fastsam-{safe}"


def _publish_start(basename, parsed, job_id):
    """Send a human-readable 'analysis started' email via SNS."""
    sns = boto3.client("sns")
    heights = parsed["extra_args"].split("--person_height", 1)
    trim = ""
    if parsed["trim_start"] or parsed["trim_end"]:
        trim = f"{parsed['trim_start'] or '0'}s → {parsed['trim_end'] or 'fin'}s"
    lines = [
        "Nouvelle analyse vidéo SAM3D lancée 🚀",
        "",
        f"Fichier       : {basename}",
        f"Job ID        : {job_id}",
        f"Args parsés   : {parsed['extra_args']}",
    ]
    if trim:
        lines.append(f"Trim          : {trim}")
    lines += [
        "",
        "L'instance GPU (g4dn.xlarge Spot) est en train de démarrer.",
        "Temps attendu : ~10-15 min (cold start inclus) pour une vidéo courte.",
        "",
        "Un email récapitulatif sera envoyé à la fin du traitement.",
    ]
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"[SAM3D] 🚀 Démarrée — {parsed['raw_name']}"[:100],
        Message="\n".join(lines),
    )


def lambda_handler(event, context):
    assert boto3 is not None, "boto3 required in Lambda runtime"
    batch = boto3.client("batch")

    submitted = []
    rejected = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        basename = os.path.basename(key)

        try:
            parsed = parse_filename(basename)
        except FilenameParseError as err:
            print(f"REJECT {key}: {err}")
            rejected.append({"key": key, "reason": str(err)})
            continue

        s3_input = f"s3://{bucket}/{key}"
        job_name = _job_name(parsed["raw_name"])

        print(f"SUBMIT {s3_input} -> {job_name}")
        print(f"  EXTRA_ARGS={parsed['extra_args']}")
        print(f"  TRIM_START={parsed['trim_start']} TRIM_END={parsed['trim_end']}")

        env_overrides = [
            {"name": "S3_INPUT_URI", "value": s3_input},
            {"name": "EXTRA_ARGS", "value": parsed["extra_args"]},
        ]
        if parsed["trim_start"]:
            env_overrides.append({"name": "TRIM_START", "value": parsed["trim_start"]})
        if parsed["trim_end"]:
            env_overrides.append({"name": "TRIM_END", "value": parsed["trim_end"]})

        response = batch.submit_job(
            jobName=job_name,
            jobQueue=JOB_QUEUE,
            jobDefinition=JOB_DEFINITION,
            containerOverrides={"environment": env_overrides},
        )
        submitted.append({
            "jobId": response["jobId"],
            "jobName": response["jobName"],
            "input": s3_input,
        })

        # Fire-and-forget "analysis started" email. Never fail the Lambda if
        # SNS is misconfigured — the job has already been queued.
        if SNS_TOPIC_ARN:
            try:
                _publish_start(basename, parsed, response["jobId"])
            except Exception as err:
                print(f"WARN: SNS publish (start) failed: {err}")

    return {
        "statusCode": 200,
        "body": json.dumps({"submitted": submitted, "rejected": rejected}),
    }


# =============================================================================
# Self-tests — run with: python aws/lambda_trigger.py
# =============================================================================
def _self_test():
    cases_ok = [
        ("squat_jean__h185.mp4",
         {"raw_name": "squat_jean",
          "extra_args": "--person_height 1.85",
          "trim_start": "", "trim_end": ""}),
        ("cmj__h185_st_com.mp4",
         {"raw_name": "cmj",
          "extra_args": "--person_height 1.85 --stationary --compute_com",
          "trim_start": "", "trim_end": ""}),
        ("violon__h185_s3_e12.mp4",
         {"raw_name": "violon",
          "extra_args": "--person_height 1.85",
          "trim_start": "3", "trim_end": "12"}),
        ("groupe__h185-170-180.mp4",
         {"raw_name": "groupe",
          "extra_args": "--multi_person --person_heights 1.85,1.70,1.80 --run_ik_per_person --write_combined_trc",
          "trim_start": "", "trim_end": ""}),
        ("grp__h185-170_st_com.mp4",
         {"raw_name": "grp",
          "extra_args": "--multi_person --person_heights 1.85,1.70 --run_ik_per_person --write_combined_trc --stationary --compute_com",
          "trim_start": "", "trim_end": ""}),
        ("Squat.MP4__h185_e5.MOV", None),  # tests uppercase ext handling
    ]
    # Last one just validates uppercase ext → let's just check it doesn't raise
    cases_reject = [
        ("squat_jean.mp4", "Missing '__<meta>'"),
        ("foo__st.mp4", "Missing required token 'h<cm>'"),
        ("foo__h185_xyz.mp4", "Unknown token"),
        ("foo__h185_h170.mp4", "Duplicate height"),
        ("foo__h185_s5_e5.mp4", "must be <"),
        ("foo__h185_s8_e5.mp4", "must be <"),
        ("foo__h185.exe", "Unsupported extension"),
        ("__h185.mp4", "Empty base name"),
        ("foo__.mp4", "Empty meta block"),
    ]

    ok = 0
    fail = 0
    for fname, expected in cases_ok:
        try:
            got = parse_filename(fname)
            if expected is None:
                print(f"OK    [{fname}] -> {got}")
                ok += 1
                continue
            for k, v in expected.items():
                assert got[k] == v, f"{k}: expected {v!r}, got {got[k]!r}"
            print(f"OK    [{fname}]")
            ok += 1
        except AssertionError as e:
            print(f"FAIL  [{fname}] -> {got} ({e})")
            fail += 1
        except FilenameParseError as e:
            print(f"FAIL  [{fname}] raised: {e}")
            fail += 1

    for fname, must_contain in cases_reject:
        try:
            parse_filename(fname)
            print(f"FAIL  [{fname}] should have been rejected")
            fail += 1
        except FilenameParseError as e:
            if must_contain in str(e):
                print(f"OK    [{fname}] rejected: {e}")
                ok += 1
            else:
                print(f"FAIL  [{fname}] wrong reason: {e} (wanted '{must_contain}')")
                fail += 1

    print(f"\n=== {ok} ok, {fail} fail ===")
    return fail == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--parse":
        # Usage: python lambda_trigger.py --parse <filename>
        print(json.dumps(parse_filename(sys.argv[2]), indent=2))
    else:
        sys.exit(0 if _self_test() else 1)
