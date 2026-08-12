"""
Lambda function that triggers an AWS Batch job when a video lands in S3.

Triggered by S3 event on s3://data-synchro-video/01-input-SAM3D/*.

Filename convention (strict — files not matching are rejected):

    <nom_libre>__<meta_tokens>.<ext>

Tokens:
    h<cm>                required, e.g. h185 -> person_height=1.85m
    h<cm>-<cm>[-...]     multi-person, e.g. h185-170 -> 2 persons 1.85m + 1.70m
    m<kg>                masse sujet en kg (e.g. m75). Optionnel mais requis
                         pour analytics auto (--module).
    a<age>               âge sujet en années (e.g. a30). Requis pour STS
                         (normes stratifiées par âge).
    sxM / sxF            sexe biologique (default: non renseigné). Pour
                         normes stratifiées.
    tr / re / cl / el    niveau : trained, recreational, clinical, elite
                         (default: trained).
    s<sec>               trim start in seconds (integer, default 0)
    e<sec>               trim end   in seconds (integer, default video end)

Module tokens (activent l'auto-analytics post-SAM3D) :
    run                  --module d3.running
    gait                 --module d3.gait
    sprint               --module d3.sprint_start
    squat                --module d3.squat
    cmj / jump           --module d3.jump
    sts                  --module d3.sit_to_stand
    cycling              --module d3.cycling
    slh                  --module d3.single_leg_hop   (RTS post-LCA, LSI)
    sls                  --module d3.single_leg_squat (RTS valgus unipodal)

Tokens RTS (tests unipodaux — une vidéo = une jambe, LSI agrégé côté app) :
    single|triple|crossover|timed6m   type de saut (slh) : --hop_type
    legR | legL          jambe testée (slh/sls) : --leg
    Ex: hop_marie__h172_m64_sxF_slh_triple_legR.mp4  → triple hop jambe D

Flags avancés (auto-dispatch par --module rend ces flags souvent redondants) :
    st                   --stationary   (auto pour cmj/squat/sts/cycling)
    com                  --compute_com
    floor                --floor        (mode aggressive : per-frame ground align)
    lv                   --lock-vertical
    bikefit              MACRO = st + lv + no_lean_fix (home-trainer)
    camR | camL          --camera_side (module cycling) : côté près caméra en
                         vue 3/4 (neutralise le membre occulté). Vide = auto.
    tt|road|comfort      --cycling_position (module cycling) : bascule les normes
                         coude/tronc/épaule/aéro. road = défaut (course cocottes/
                         drops), tt = contre-la-montre (coude ~90-105°, CdA bas =
                         optimal), comfort = position droite.

Examples:
    squat_jean__h185_m85_a30_squat.mp4                     → auto d3.squat
    cmj_athlete__h180_m75_a25_sxF_tr_cmj.mp4               → auto d3.jump
    running_cedric__h178_m70_a35_sxM_run.mp4               → auto d3.running
    sts_diane__h165_m82_a65_sxF_cl_sts.mp4                 → auto d3.sit_to_stand
    bikefit_alex__h180_m75_a40_bikefit_cycling.mp4         → bikefit + cycling
    titia_clm__h165_m55_a23_sxF_el_cycling_tt.mp4          → cycling position CLM
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
MASS_RE = re.compile(r"^m(\d{2,3})$")
AGE_RE = re.compile(r"^a(\d{1,3})$")
SEX_RE = re.compile(r"^sx([MF])$")
START_RE = re.compile(r"^s(\d+)$")
END_RE = re.compile(r"^e(\d+)$")
TREADMILL_RE = re.compile(r"^tm(\d+)$")  # vitesse tapis en km/h (tm12 = 12 km/h)
FLAG_TOKENS = {"st", "com", "floor", "lv", "bikefit", "ca", "seated"}
LEVEL_TOKENS = {"tr": "trained", "re": "recreational",
                "cl": "clinical", "el": "elite"}
# Module token → --module value (d3 par défaut pour SAM3D 3D pipeline)
HOP_TYPE_TOKENS = {"single", "triple", "crossover", "timed6m"}  # single_leg_hop
MODULE_TOKENS = {
    "run": "d3.running",
    "gait": "d3.gait",
    "sprint": "d3.sprint_start",
    "squat": "d3.squat",
    "slh": "d3.single_leg_hop",     # RTS : single leg hop (LSI)
    "sls": "d3.single_leg_squat",   # RTS : single leg squat (valgus)
    "cmj": "d3.jump",
    "jump": "d3.jump",
    "sts": "d3.sit_to_stand",
    "cycling": "d3.cycling",
}

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
    mass_kg = None
    age = None
    sex = None
    level = None
    module = None
    treadmill_mps = None
    trim_start = None
    trim_end = None
    stationary = False
    compute_com = False
    floor = False
    lock_vertical = False
    bikefit = False
    contact_anchor = False  # token 'ca' → --contact_anchor
    floor_seated = False    # token 'seated' → --floor_seated
    cycling_position = None  # road (défaut) / tt / comfort → stratum normes vélo
    camera_side = None       # camR / camL → --camera_side (vue 3/4 cyclisme)
    hop_type = None          # single/triple/crossover/timed6m → single_leg_hop RTS
    leg = None               # legR / legL → jambe testée (tests unipodaux RTS)

    for t in tokens:
        m = HEIGHT_RE.match(t)
        if m:
            if heights is not None:
                raise FilenameParseError(f"Duplicate height token: '{t}'")
            heights = [int(v) / 100.0 for v in m.group(1).split("-")]
            continue
        m = MASS_RE.match(t)
        if m:
            if mass_kg is not None:
                raise FilenameParseError(f"Duplicate mass token: '{t}'")
            mass_kg = int(m.group(1))
            continue
        m = AGE_RE.match(t)
        if m:
            if age is not None:
                raise FilenameParseError(f"Duplicate age token: '{t}'")
            age = int(m.group(1))
            continue
        m = SEX_RE.match(t)
        if m:
            if sex is not None:
                raise FilenameParseError(f"Duplicate sex token: '{t}'")
            sex = m.group(1)
            continue
        m = TREADMILL_RE.match(t)
        if m:
            if treadmill_mps is not None:
                raise FilenameParseError(f"Duplicate treadmill token: '{t}'")
            # km/h → m/s (÷3.6). tm12 = 12 km/h = 3.33 m/s.
            treadmill_mps = round(int(m.group(1)) / 3.6, 2)
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
        if t in LEVEL_TOKENS:
            if level is not None:
                raise FilenameParseError(f"Duplicate level token: '{t}'")
            level = LEVEL_TOKENS[t]
            continue
        if t in MODULE_TOKENS:
            if module is not None:
                raise FilenameParseError(f"Duplicate module token: '{t}'")
            module = MODULE_TOKENS[t]
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
        if t == "lv":
            lock_vertical = True
            continue
        if t == "bikefit":
            bikefit = True
            continue
        if t == "seated":
            # Sujet assis : mise au sol per-frame SANS le redressement
            # "corps vertical", qui suppose un sujet debout.
            floor_seated = True
            continue
        if t == "ca":
            # Ancrage sol conscient du contact. Remplace l'ancien --feet_anchor
            # pour les exercices : re-ancre le pied en appui soutenu (le bassin
            # descend bien dans un squat) tout en preservant les phases de vol.
            contact_anchor = True
            continue
        if t in ("tt", "road", "comfort"):
            if cycling_position is not None:
                raise FilenameParseError(f"Duplicate cycling position token: '{t}'")
            cycling_position = t
            continue
        if t in ("camR", "camL"):
            if camera_side is not None:
                raise FilenameParseError(f"Duplicate camera side token: '{t}'")
            camera_side = t[-1]  # R / L
            continue
        if t in HOP_TYPE_TOKENS:
            if hop_type is not None:
                raise FilenameParseError(f"Duplicate hop type token: '{t}'")
            hop_type = t
            continue
        if t in ("legR", "legL"):
            if leg is not None:
                raise FilenameParseError(f"Duplicate leg token: '{t}'")
            leg = t[-1]  # R / L
            continue
        raise FilenameParseError(f"Unknown token: '{t}'")

    if heights is None:
        raise FilenameParseError("Missing required token 'h<cm>'")
    if trim_start is not None and trim_end is not None and trim_start >= trim_end:
        raise FilenameParseError(
            f"start ({trim_start}s) must be < end ({trim_end}s)"
        )
    # Macro `bikefit` = --stationary + --lock-vertical + --no_lean_fix.
    # Développée AVANT la sérialisation en flags CLI.
    if bikefit:
        stationary = True
        lock_vertical = True
        if floor:
            raise FilenameParseError(
                "Token 'bikefit' incompatible avec 'floor' — sur home-trainer "
                "il n'y a pas de sol libre à détecter."
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
    if floor_seated:
        extra.append("--floor_seated")
    if contact_anchor:
        extra.append("--contact_anchor")
    if lock_vertical:
        extra.append("--lock-vertical")
    if bikefit:
        # Macro : ajoute --no_lean_fix (les autres flags ont déjà été activés
        # dans la section normalisation ci-dessus).
        extra.append("--no_lean_fix")

    # Auto-analytics : ajoute --module + --mass_kg + --age + --sex + --level
    # si le nom de fichier fournit ces tokens. Le pipeline SAM3D relaie ces
    # args à synkro-analytics en post-processing.
    if module is not None:
        if mass_kg is None:
            raise FilenameParseError(
                f"Module token '{module}' nécessite m<kg> (ex: m75) pour analytics."
            )
        extra += ["--module", module,
                   "--mass_kg", str(mass_kg)]
        if age is not None:
            extra += ["--age", str(age)]
        if sex is not None:
            extra += ["--sex", sex]
        if level is not None:
            extra += ["--level", level]
        if treadmill_mps is not None:
            extra += ["--treadmill_speed", str(treadmill_mps)]
        if cycling_position is not None:
            if module != "d3.cycling":
                raise FilenameParseError(
                    f"Token position '{cycling_position}' réservé au module cycling."
                )
            extra += ["--cycling_position", cycling_position]
        if camera_side is not None:
            if module != "d3.cycling":
                raise FilenameParseError(
                    f"Token 'cam{camera_side}' réservé au module cycling."
                )
            extra += ["--camera_side", camera_side]
        if hop_type is not None:
            if module != "d3.single_leg_hop":
                raise FilenameParseError(
                    f"Token hop '{hop_type}' réservé au module 'slh' (single_leg_hop)."
                )
            extra += ["--hop_type", hop_type]
        if leg is not None:
            if module not in ("d3.single_leg_hop", "d3.single_leg_squat"):
                raise FilenameParseError(
                    f"Token 'leg{leg}' réservé aux tests unipodaux (slh/sls)."
                )
            extra += ["--leg", leg]
    elif cycling_position is not None:
        raise FilenameParseError(
            f"Token position '{cycling_position}' nécessite le module 'cycling'."
        )
    elif camera_side is not None:
        raise FilenameParseError(
            f"Token 'cam{camera_side}' nécessite le module 'cycling'."
        )
    elif hop_type is not None or leg is not None:
        raise FilenameParseError(
            "Tokens hop/leg nécessitent un module RTS (slh/sls)."
        )

    # Toujours activer --floor_moge (fix Y-DOWN 2026-07 : marche pour tous les
    # cas standard, auto-skip si MoGe échoue).
    if "--floor_moge" not in extra and not floor:
        extra.append("--floor_moge")

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
          "extra_args": "--person_height 1.85 --floor_moge",
          "trim_start": "", "trim_end": ""}),
        ("cmj__h185_st_com.mp4",
         {"raw_name": "cmj",
          "extra_args": "--person_height 1.85 --stationary --compute_com --floor_moge",
          "trim_start": "", "trim_end": ""}),
        ("violon__h185_s3_e12.mp4",
         {"raw_name": "violon",
          "extra_args": "--person_height 1.85 --floor_moge",
          "trim_start": "3", "trim_end": "12"}),
        ("groupe__h185-170-180.mp4",
         {"raw_name": "groupe",
          "extra_args": "--multi_person --person_heights 1.85,1.70,1.80 --run_ik_per_person --write_combined_trc --floor_moge",
          "trim_start": "", "trim_end": ""}),
        ("grp__h185-170_st_com.mp4",
         {"raw_name": "grp",
          "extra_args": "--multi_person --person_heights 1.85,1.70 --run_ik_per_person --write_combined_trc --stationary --compute_com --floor_moge",
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
