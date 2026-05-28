#!/bin/bash
# Batch generation of avatars for a folder of exercise videos.
# Runs generate_avatars.py via Docker for each video found in INPUT_DIR.
#
# Usage:
#   INPUT_DIR=./videos_to_batch HEIGHT=1.75 ./scripts/batch_avatars_local.sh
#
# Output structure:
#   outputs/avatars/<exercise_name>/
#     ├── markers_<exercise>.trc
#     ├── markers_<exercise>_avatar_female_young.glb
#     ├── markers_<exercise>_avatar_male_young.glb
#     └── ... (1 par template présent dans assets/avatars/)
set -e

INPUT_DIR="${INPUT_DIR:-./videos_to_batch}"
HEIGHT="${HEIGHT:-1.75}"
EXTRA_ARGS="${EXTRA_ARGS:---floor}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "ERROR: INPUT_DIR=$INPUT_DIR not found"
  echo "Create it and drop your exercise videos inside, then re-run."
  exit 1
fi

shopt -s nullglob nocaseglob
videos=("$INPUT_DIR"/*.{mp4,mov,avi})
shopt -u nullglob nocaseglob

if [ ${#videos[@]} -eq 0 ]; then
  echo "ERROR: no video found in $INPUT_DIR (looking for *.mp4, *.mov, *.avi)"
  exit 1
fi

echo "=== Batch avatar generation ==="
echo "  Input dir:  $INPUT_DIR"
echo "  Height:     $HEIGHT m"
echo "  Extra args: $EXTRA_ARGS"
echo "  Videos:     ${#videos[@]}"
echo ""

# Place INPUT_DIR symlink inside ./videos so Docker can mount it (videos is the
# only mounted folder in docker-compose.yml).
mkdir -p ./videos
for video in "${videos[@]}"; do
  filename=$(basename "$video")
  if [ ! -e "./videos/$filename" ]; then
    cp "$video" "./videos/$filename"
  fi
done

mkdir -p ./outputs/avatars

for video in "${videos[@]}"; do
  filename=$(basename "$video")
  exercise=$(basename "$filename" | sed 's/\.[^.]*$//')
  output_name="avatars/$exercise"

  echo ""
  echo "--- [$exercise] ---"
  docker compose run --rm fast-sam-opensim python generate_avatars.py \
      --video_path "/app/videos/$filename" \
      --output_dir "/outputs/$output_name" \
      --inference_type body \
      --markerset flodelaplace \
      --floor_moge \
      --detector_model checkpoints/yolo/yolo11m-pose.engine \
      --bbox_thr 0.2 --nms_thr 0.9 \
      --detect_then_infer --inference_batch_cap 4 \
      --fallback_lower_bbox 0.05 --fallback_nms 0.9 --fallback_iou_thresh 0.5 \
      --person_height "$HEIGHT" \
      $EXTRA_ARGS
done

echo ""
echo "=== Done ==="
echo "  Avatars under: ./outputs/avatars/"
ls -1 ./outputs/avatars/
