"""
Lambda function that triggers an AWS Batch AVATAR job when a video lands in S3.

Triggered by S3 event on s3://data-synchro-video/01-input-avatar/<user>/<exercise>__<meta>.<ext>.

S3 key convention (strict):
    01-input-avatar/<user_id>/<exercise_id>__<meta_tokens>.<ext>

The lambda :
  - parses user_id from the S3 key (the directory right under 01-input-avatar/)
  - parses exercise_id from the filename (before "__")
  - parses meta tokens (height + flags)
  - submits a Batch job with MODE=avatar
  - sets S3_OUTPUT_URI = s3://<bucket>/02-output-avatar/<user_id>/<exercise_id>/
  - the container will then upload {template}.glb directly under that prefix.

Filename meta tokens (post `__`):
    h<cm>      required, single subject height in cm (e.g. h180 -> 1.80m)
    s<sec>     trim start in seconds
    e<sec>     trim end in seconds
    floor      --floor   (sujet debout : squat, marche, fente, etc.)
    seated     --floor_seated  (sujet assis : 5STS, chaise, Lasègue)
    anchor     --feet_anchor   (verrouille position XZ des pieds, pour pieds-au-sol)
    st         --stationary
    com        --compute_com   (rarement utile pour avatar)

Multi-person is NOT supported in avatar mode (1 video = 1 subject = 4 avatars).

Examples:
    01-input-avatar/kine_42/squat_001__h180_floor.mp4
        → submit avec --person_height 1.80 --floor
        → output: s3://.../02-output-avatar/kine_42/squat_001/{female_young,male_young,...}.glb

    01-input-avatar/kine_42/5sts_001__h165_seated_anchor.mp4
        → submit avec --person_height 1.65 --floor_seated --feet_anchor
        → output: s3://.../02-output-avatar/kine_42/5sts_001/

    01-input-avatar/kine_42/rameur_001__h175.mp4
        → submit avec --person_height 1.75  (pas de floor — rameur assis machine)
        → output: s3://.../02-output-avatar/kine_42/rameur_001/
"""
import json
import os
import re
import sys
import urllib.parse

try:
    import boto3
except ImportError:
    boto3 = None  # local unit tests

JOB_QUEUE = os.environ.get("JOB_QUEUE", "synkro-fastsam3d-queue")
JOB_DEFINITION = os.environ.get("JOB_DEFINITION", "synkro-fastsam3d-avatar-job")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
S3_OUTPUT_BUCKET = os.environ.get("S3_OUTPUT_BUCKET", "data-synchro-video")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "02-output-avatar")
S3_INPUT_PREFIX = os.environ.get("S3_INPUT_PREFIX", "01-input-avatar")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

HEIGHT_RE = re.compile(r"^h(\d{2,3})$")
START_RE = re.compile(r"^s(\d+)$")
END_RE = re.compile(r"^e(\d+)$")
FLAG_TOKENS = {"st", "com", "floor", "seated", "anchor"}

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


class FilenameParseError(ValueError):
    pass


def parse_key(key):
    """
    Parse the S3 key into (user_id, exercise_id, parsed_meta).

    Expected layout: <S3_INPUT_PREFIX>/<user_id>/<exercise_id>__<meta>.<ext>
    """
    if not key.startswith(S3_INPUT_PREFIX + "/"):
        raise FilenameParseError(
            f"Key must start with '{S3_INPUT_PREFIX}/' (got {key!r})"
        )
    rest = key[len(S3_INPUT_PREFIX) + 1:]
    parts = rest.split("/")
    if len(parts) < 2:
        raise FilenameParseError(
            f"Key must be '{S3_INPUT_PREFIX}/<user_id>/<exercise_id>__<meta>.<ext>' "
            f"(got {key!r})"
        )
    user_id = parts[0]
    basename = parts[-1]
    if not user_id:
        raise FilenameParseError("Empty user_id")
    parsed = parse_filename(basename)
    return user_id, parsed


def parse_filename(basename):
    """
    Parse a video filename into AWS Batch env-var overrides.

    Returns dict:
        exercise_id: original name without extension and without meta (== raw_name)
        extra_args:  string of flags for generate_avatars.py
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
            f"Expected: <exercise_id>__h<cm>[_floor|_seated|_anchor|_st|_com|_s<sec>|_e<sec>]{ext}"
        )

    exercise_id, _, meta_block = name.rpartition("__")
    if not exercise_id:
        raise FilenameParseError("Empty exercise_id before '__'")
    tokens = [t for t in meta_block.split("_") if t]
    if not tokens:
        raise FilenameParseError("Empty meta block after '__'")

    height_cm = None
    trim_start = None
    trim_end = None
    stationary = False
    compute_com = False
    floor = False
    seated = False
    anchor = False

    for t in tokens:
        m = HEIGHT_RE.match(t)
        if m:
            if height_cm is not None:
                raise FilenameParseError(f"Duplicate height token: '{t}'")
            height_cm = int(m.group(1))
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
        if t == "seated":
            seated = True
            continue
        if t == "anchor":
            anchor = True
            continue
        raise FilenameParseError(f"Unknown token: '{t}'")

    if height_cm is None:
        raise FilenameParseError("Missing required token 'h<cm>'")
    if trim_start is not None and trim_end is not None and trim_start >= trim_end:
        raise FilenameParseError(
            f"start ({trim_start}s) must be < end ({trim_end}s)"
        )
    if floor and seated:
        raise FilenameParseError(
            "Tokens 'floor' and 'seated' are mutually exclusive "
            "('seated' implique déjà --floor_seated)."
        )

    extra = ["--person_height", f"{height_cm / 100.0:.2f}"]
    if stationary:
        extra.append("--stationary")
    if compute_com:
        extra.append("--compute_com")
    if floor:
        extra.append("--floor")
    if seated:
        extra.append("--floor_seated")
    if anchor:
        extra.append("--feet_anchor")

    return {
        "exercise_id": exercise_id,
        "extra_args": " ".join(extra),
        "trim_start": str(trim_start) if trim_start is not None else "",
        "trim_end": str(trim_end) if trim_end is not None else "",
    }


def _job_name(user_id, exercise_id):
    safe_u = _SANITIZE_RE.sub("-", user_id)[:40] or "user"
    safe_e = _SANITIZE_RE.sub("-", exercise_id)[:60] or "ex"
    return f"avatar-{safe_u}-{safe_e}"[:128]


def _publish_start(basename, user_id, parsed, job_id):
    """Send 'avatar job started' email via SNS."""
    sns = boto3.client("sns")
    trim = ""
    if parsed["trim_start"] or parsed["trim_end"]:
        trim = f"{parsed['trim_start'] or '0'}s → {parsed['trim_end'] or 'fin'}s"
    lines = [
        "Nouvelle génération d'avatars SAM3D lancée 🤖",
        "",
        f"User          : {user_id}",
        f"Exercise      : {parsed['exercise_id']}",
        f"Fichier       : {basename}",
        f"Job ID        : {job_id}",
        f"Args parsés   : {parsed['extra_args']}",
    ]
    if trim:
        lines.append(f"Trim          : {trim}")
    lines += [
        "",
        "L'instance GPU (g4dn.xlarge Spot) traite la vidéo et génère un GLB par",
        "template d'avatar (homme/femme jeune/âgé) à la fin.",
        "",
        "Un email récapitulatif sera envoyé en fin de job.",
    ]
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"[AVATAR] 🚀 Démarrée — {user_id}/{parsed['exercise_id']}"[:100],
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
            user_id, parsed = parse_key(key)
        except FilenameParseError as err:
            print(f"REJECT {key}: {err}")
            rejected.append({"key": key, "reason": str(err)})
            continue

        s3_input = f"s3://{bucket}/{key}"
        s3_output = (
            f"s3://{S3_OUTPUT_BUCKET}/{S3_OUTPUT_PREFIX}/"
            f"{user_id}/{parsed['exercise_id']}/"
        )
        job_name = _job_name(user_id, parsed["exercise_id"])

        print(f"SUBMIT {s3_input} -> {job_name}")
        print(f"  OUTPUT={s3_output}")
        print(f"  EXTRA_ARGS={parsed['extra_args']}")
        print(f"  TRIM_START={parsed['trim_start']} TRIM_END={parsed['trim_end']}")

        env_overrides = [
            {"name": "MODE", "value": "avatar"},
            {"name": "S3_INPUT_URI", "value": s3_input},
            {"name": "S3_OUTPUT_URI", "value": s3_output},
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
            "output": s3_output,
        })

        if SNS_TOPIC_ARN:
            try:
                _publish_start(basename, user_id, parsed, response["jobId"])
            except Exception as err:
                print(f"WARN: SNS publish (start) failed: {err}")

    return {
        "statusCode": 200,
        "body": json.dumps({"submitted": submitted, "rejected": rejected}),
    }


# =============================================================================
# Self-tests — run with: python aws/lambda_trigger_avatar.py
# =============================================================================
def _self_test():
    cases_ok_filename = [
        ("squat_001__h180_floor.mp4",
         {"exercise_id": "squat_001",
          "extra_args": "--person_height 1.80 --floor",
          "trim_start": "", "trim_end": ""}),
        ("5sts_001__h165_seated_anchor.mp4",
         {"exercise_id": "5sts_001",
          "extra_args": "--person_height 1.65 --floor_seated --feet_anchor",
          "trim_start": "", "trim_end": ""}),
        ("rameur__h175.mp4",
         {"exercise_id": "rameur",
          "extra_args": "--person_height 1.75",
          "trim_start": "", "trim_end": ""}),
        ("cmj__h185_st_floor.mp4",
         {"exercise_id": "cmj",
          "extra_args": "--person_height 1.85 --stationary --floor",
          "trim_start": "", "trim_end": ""}),
        ("violon__h172_s3_e15.mp4",
         {"exercise_id": "violon",
          "extra_args": "--person_height 1.72",
          "trim_start": "3", "trim_end": "15"}),
    ]
    cases_reject_filename = [
        ("foo.mp4", "Missing '__<meta>'"),
        ("foo__.mp4", "Empty meta block"),
        ("foo__st.mp4", "Missing required token 'h<cm>'"),
        ("foo__h180_xyz.mp4", "Unknown token"),
        ("foo__h180_h170.mp4", "Duplicate height"),
        ("foo__h180_s5_e5.mp4", "must be <"),
        ("foo__h180_floor_seated.mp4", "mutually exclusive"),
        ("__h180.mp4", "Empty exercise_id"),
    ]
    cases_ok_key = [
        ("01-input-avatar/kine_42/squat_001__h180_floor.mp4",
         "kine_42", "squat_001"),
        ("01-input-avatar/u/ex__h180.mov",
         "u", "ex"),
    ]
    cases_reject_key = [
        ("squat__h180.mp4", "must start with"),
        ("01-input-avatar/squat__h180.mp4", "user_id"),
        ("01-input-avatar//squat__h180.mp4", "Empty user_id"),
    ]

    ok = 0
    fail = 0
    for fname, expected in cases_ok_filename:
        try:
            got = parse_filename(fname)
            for k, v in expected.items():
                assert got[k] == v, f"{k}: expected {v!r}, got {got[k]!r}"
            print(f"OK    [{fname}]")
            ok += 1
        except (AssertionError, FilenameParseError) as e:
            print(f"FAIL  [{fname}] {e}")
            fail += 1

    for fname, must_contain in cases_reject_filename:
        try:
            parse_filename(fname)
            print(f"FAIL  [{fname}] should have been rejected")
            fail += 1
        except FilenameParseError as e:
            if must_contain in str(e):
                print(f"OK    [{fname}] rejected: {e}")
                ok += 1
            else:
                print(f"FAIL  [{fname}] wrong reason: {e}")
                fail += 1

    for key, exp_user, exp_ex in cases_ok_key:
        try:
            uid, parsed = parse_key(key)
            assert uid == exp_user, f"user {uid!r} != {exp_user!r}"
            assert parsed["exercise_id"] == exp_ex
            print(f"OK    [{key}] -> user={uid} ex={parsed['exercise_id']}")
            ok += 1
        except (AssertionError, FilenameParseError) as e:
            print(f"FAIL  [{key}] {e}")
            fail += 1

    for key, must_contain in cases_reject_key:
        try:
            parse_key(key)
            print(f"FAIL  [{key}] should have been rejected")
            fail += 1
        except FilenameParseError as e:
            if must_contain in str(e):
                print(f"OK    [{key}] rejected: {e}")
                ok += 1
            else:
                print(f"FAIL  [{key}] wrong reason: {e}")
                fail += 1

    print(f"\n=== {ok} ok, {fail} fail ===")
    return fail == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--parse":
        # python lambda_trigger_avatar.py --parse 01-input-avatar/u/ex__h180.mp4
        uid, parsed = parse_key(sys.argv[2])
        print(json.dumps({"user_id": uid, **parsed}, indent=2))
    else:
        sys.exit(0 if _self_test() else 1)
