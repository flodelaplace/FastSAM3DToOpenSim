#!/bin/bash
# Local Docker smoke test — runs the same inference command as your local setup
# Usage: ./scripts/test_docker_local.sh
set -e

VIDEO="${VIDEO:-22516499.mp4}"
OUTPUT_NAME="${OUTPUT_NAME:-test_docker_$(basename "$VIDEO" | sed 's/\.[^.]*$//')}"

docker compose run --rm fast-sam-opensim python demo_video_opensim.py \
    --video_path "/app/videos/${VIDEO}" \
    --output_dir "/outputs/${OUTPUT_NAME}" \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --multi_person --max_persons 6 --write_combined_trc \
    --floor_moge --max_frames 200 \
    --bbox_thr 0.2 --nms_thr 0.9 \
    --detect_then_infer --inference_batch_cap 4 \
    --fallback_lower_bbox 0.05 --fallback_nms 0.9 --fallback_iou_thresh 0.5 \
    --run_ik_per_person --person_heights 1.62,1.64,1.61,1.70,1.78,1.75
