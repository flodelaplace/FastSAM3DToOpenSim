#!/bin/bash
# Local Docker smoke test — same args as the AWS Batch job (single-person kiné use case).
# Multi-person path is exercised via EXTRA_ARGS (see examples at bottom).
#
# Usage:
#   VIDEO=Squat.MP4 HEIGHT=1.85 ./scripts/test_docker_local.sh
#   VIDEO=cmj.mp4 HEIGHT=1.75 EXTRA_ARGS="--stationary --compute_com" ./scripts/test_docker_local.sh
set -e

VIDEO="${VIDEO:-Squat.MP4}"
HEIGHT="${HEIGHT:-1.85}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUTPUT_NAME="${OUTPUT_NAME:-test_docker_$(basename "$VIDEO" | sed 's/\.[^.]*$//')}"

echo "=== Local Docker smoke test ==="
echo "  Video:   ./videos/$VIDEO"
echo "  Height:  $HEIGHT m"
echo "  Extra:   $EXTRA_ARGS"
echo "  Output:  ./outputs/$OUTPUT_NAME"
echo ""

# Assure que docker-compose écrit les outputs avec ton UID:GID (voir docker-compose.yml)
export DOCKER_UID="$(id -u)" DOCKER_GID="$(id -g)"

# shellcheck disable=SC2086
docker compose run --rm fast-sam-opensim python demo_video_opensim.py \
    --video_path "/app/videos/${VIDEO}" \
    --output_dir "/outputs/${OUTPUT_NAME}" \
    --inference_type body \
    --markerset flodelaplace \
    --floor_moge \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --bbox_thr 0.2 --nms_thr 0.9 \
    --detect_then_infer --inference_batch_cap 4 \
    --fallback_lower_bbox 0.05 --fallback_nms 0.9 --fallback_iou_thresh 0.5 \
    --person_height "$HEIGHT" \
    $EXTRA_ARGS

# -----------------------------------------------------------------------------
# Examples of EXTRA_ARGS (mirror what the AWS Lambda builds from filenames):
#
#   stationary + CoM:
#     EXTRA_ARGS="--stationary --compute_com"
#
#   multi-person (3 subjects):
#     EXTRA_ARGS="--multi_person --person_heights 1.85,1.70,1.80 \
#                 --run_ik_per_person --write_combined_trc"
#
#   (trim is NOT tested here — that's done by scripts/run_job.sh via ffmpeg)
# -----------------------------------------------------------------------------
