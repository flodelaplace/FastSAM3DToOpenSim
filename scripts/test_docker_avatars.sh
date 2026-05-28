#!/bin/bash
# Local Docker smoke test for the avatar-only pipeline.
# Same args as the SAM3D one (test_docker_local.sh) but invokes generate_avatars.py.
#
# Usage:
#   VIDEO=Squat.MP4 HEIGHT=1.85 ./scripts/test_docker_avatars.sh
#   VIDEO=YogaEla.mp4 HEIGHT=1.65 EXTRA_ARGS="--floor" ./scripts/test_docker_avatars.sh
set -e

VIDEO="${VIDEO:-Squat.MP4}"
HEIGHT="${HEIGHT:-1.75}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUTPUT_NAME="${OUTPUT_NAME:-avatars_$(basename "$VIDEO" | sed 's/\.[^.]*$//')}"

echo "=== Local Docker avatar generation ==="
echo "  Video:   ./videos/$VIDEO"
echo "  Height:  $HEIGHT m"
echo "  Extra:   $EXTRA_ARGS"
echo "  Output:  ./outputs/$OUTPUT_NAME"
echo ""

# shellcheck disable=SC2086
docker compose run --rm fast-sam-opensim python generate_avatars.py \
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
