#!/bin/bash
# =============================================================================
# run_job.sh — AWS Batch job wrapper
#
# Expected environment variables:
#   S3_INPUT_URI         s3://bucket/input/video.mp4            (required)
#   S3_OUTPUT_URI        s3://bucket/output/                     (required, trailing /)
#   CHECKPOINTS_S3_URI   s3://bucket/checkpoints/                (required, trailing /)
#   INFERENCE_TYPE       body | full                             (default: body)
#   EXTRA_ARGS           per-video flags from filename parser    (default: empty)
#   TRIM_START           integer seconds (ffmpeg -ss)            (optional)
#   TRIM_END             integer seconds (ffmpeg -to)            (optional)
#
# Flow: sync checkpoints → pull video → [ffmpeg trim] → process → push to S3
# =============================================================================
set -euo pipefail

# ---- Validate required vars -------------------------------------------------
: "${S3_INPUT_URI:?S3_INPUT_URI is not set}"
: "${S3_OUTPUT_URI:?S3_OUTPUT_URI is not set}"
: "${CHECKPOINTS_S3_URI:?CHECKPOINTS_S3_URI is not set}"

INFERENCE_TYPE="${INFERENCE_TYPE:-body}"
VIDEO_BASENAME=$(basename "$S3_INPUT_URI")
VIDEO_NAME="${VIDEO_BASENAME%.*}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_INPUT="/tmp/input/${VIDEO_BASENAME}"
PROC_INPUT="$LOCAL_INPUT"
OUTPUT_DIR="/outputs/output_${TIMESTAMP}_${VIDEO_NAME}"
S3_OUTPUT_PATH="${S3_OUTPUT_URI%/}/output_${TIMESTAMP}_${VIDEO_NAME}/"

echo "=== FastSAM3DToOpenSim AWS Batch Job ==="
echo "  Input:         $S3_INPUT_URI"
echo "  Output S3:     $S3_OUTPUT_PATH"
echo "  Checkpoints:   $CHECKPOINTS_S3_URI"
echo "  Inference:     $INFERENCE_TYPE"
echo "  Extra args:    ${EXTRA_ARGS:-<none>}"
echo "  Trim:          ${TRIM_START:-0}s -> ${TRIM_END:-<end>}"

# ---- Sync checkpoints from S3 -----------------------------------------------
echo ">>> Syncing checkpoints from S3..."
mkdir -p /app/checkpoints
aws s3 sync "$CHECKPOINTS_S3_URI" /app/checkpoints/ --no-progress

# ---- Pull video from S3 -----------------------------------------------------
echo ">>> Downloading video from S3..."
mkdir -p /tmp/input
aws s3 cp "$S3_INPUT_URI" "$LOCAL_INPUT" --no-progress

# ---- Optional ffmpeg trim (accurate, re-encoded) ----------------------------
if [ -n "${TRIM_START:-}" ] || [ -n "${TRIM_END:-}" ]; then
    TRIMMED="/tmp/input/trimmed_${VIDEO_BASENAME}"
    START="${TRIM_START:-0}"
    FFMPEG_ARGS=(-y -ss "$START" -i "$LOCAL_INPUT")
    if [ -n "${TRIM_END:-}" ]; then
        DURATION=$((TRIM_END - START))
        FFMPEG_ARGS+=(-t "$DURATION")
    fi
    FFMPEG_ARGS+=(-c:v libx264 -preset veryfast -crf 18 -an "$TRIMMED")
    echo ">>> Trimming video: ffmpeg ${FFMPEG_ARGS[*]}"
    ffmpeg "${FFMPEG_ARGS[@]}"
    PROC_INPUT="$TRIMMED"
fi

# ---- Activate conda env -----------------------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body

# ---- Build TRT engines if missing, then cache them back to S3 ---------------
# Engines are portable across g4dn.xlarge instances (all use T4 / SM 7.5) and
# across the same Docker image (TRT version pinned). The initial sync above
# already pulled them if present on S3; we only build what's missing, then push
# the freshly-built engines back so the next cold start skips the rebuild.
#
# ⚠️ If you rebuild the Docker image with a different TensorRT version, purge
# the cached engines on S3 first:
#   aws s3 rm s3://data-synchro-video/checkpoints/yolo/yolo11m-pose.engine
#   aws s3 rm s3://data-synchro-video/checkpoints/moge_trt/ --recursive --exclude "*" --include "*.engine"
#   aws s3 rm s3://data-synchro-video/checkpoints/sam-3d-body-dinov3/backbone_trt/ --recursive
cd /app
if [ ! -f "checkpoints/yolo/yolo11m-pose.engine" ]; then
    echo ">>> Building YOLO TRT engine..."
    python -c "from ultralytics import YOLO; YOLO('checkpoints/yolo/yolo11m-pose.pt').export(format='engine', device=0, half=True, imgsz=640)"
    echo ">>> Caching YOLO engine to S3..."
    aws s3 cp checkpoints/yolo/yolo11m-pose.engine \
        "${CHECKPOINTS_S3_URI%/}/yolo/yolo11m-pose.engine" --no-progress
fi
if [ ! -f "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine" ]; then
    echo ">>> Building MoGe TRT engine..."
    python convert_moge_encoder_trt.py --all
    echo ">>> Caching MoGe engines to S3..."
    aws s3 sync checkpoints/moge_trt/ "${CHECKPOINTS_S3_URI%/}/moge_trt/" \
        --exclude "*" --include "*.engine" --no-progress
fi
if [ ! -f "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine" ]; then
    echo ">>> Building DINOv3 backbone TRT engine (this may take 10-15 min)..."
    # --all runs a post-build benchmark step (Step 3) that can GPU-OOM on T4
    # (16 GB VRAM): the 1.6 GB engine + torch.compile state + batch-2 test
    # saturates memory, raising `'NoneType' has no attribute set_input_shape`.
    # The benchmark is non-essential — tolerate its failure and proceed to cache
    # as long as the .engine file was successfully written.
    USE_TRT_BACKBONE=0 python convert_backbone_tensorrt.py --all || true
    if [ ! -f "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine" ]; then
        echo ">>> ERROR: DINOv3 engine build failed (no .engine file written)"
        exit 1
    fi
    echo ">>> Caching DINOv3 backbone engines to S3..."
    aws s3 sync checkpoints/sam-3d-body-dinov3/backbone_trt/ \
        "${CHECKPOINTS_S3_URI%/}/sam-3d-body-dinov3/backbone_trt/" --no-progress
fi

# ---- Run inference ----------------------------------------------------------
# Static prod flags — per-video flags come via EXTRA_ARGS (set by the Lambda
# from the filename meta block: h<cm>, s/e trim, st, com, multi-person).
echo ">>> Processing video..."
# shellcheck disable=SC2086
python demo_video_opensim.py \
    --video_path "$PROC_INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --inference_type "$INFERENCE_TYPE" \
    --markerset flodelaplace \
    --floor_moge \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --bbox_thr 0.2 --nms_thr 0.9 \
    --detect_then_infer --inference_batch_cap 4 \
    --fallback_lower_bbox 0.05 --fallback_nms 0.9 --fallback_iou_thresh 0.5 \
    ${EXTRA_ARGS:-}

# ---- Push results to S3 -----------------------------------------------------
echo ">>> Uploading results to S3..."
aws s3 cp "$OUTPUT_DIR" "$S3_OUTPUT_PATH" --recursive --no-progress

echo "=== Job complete -> $S3_OUTPUT_PATH ==="
