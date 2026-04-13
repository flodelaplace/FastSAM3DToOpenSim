#!/bin/bash
# =============================================================================
# run_job.sh — AWS Batch job wrapper
#
# Expected environment variables (set by AWS Batch job definition):
#   S3_INPUT_URI    s3://bucket/input/video.mp4
#   S3_OUTPUT_URI   s3://bucket/output/          (trailing slash)
#   INFERENCE_TYPE  body | full                   (default: body)
#   EXTRA_ARGS      optional extra flags for demo_video_opensim.py
#
# Flow: pull video from S3 -> process -> push results to S3
# =============================================================================
set -euo pipefail

# ---- Validate inputs --------------------------------------------------------
if [ -z "${S3_INPUT_URI:-}" ]; then
    echo "ERROR: S3_INPUT_URI is not set" >&2
    exit 1
fi
if [ -z "${S3_OUTPUT_URI:-}" ]; then
    echo "ERROR: S3_OUTPUT_URI is not set" >&2
    exit 1
fi

INFERENCE_TYPE="${INFERENCE_TYPE:-body}"
VIDEO_BASENAME=$(basename "$S3_INPUT_URI")
VIDEO_NAME="${VIDEO_BASENAME%.*}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_VIDEO="/tmp/input/${VIDEO_BASENAME}"
OUTPUT_DIR="/outputs/output_${TIMESTAMP}_${VIDEO_NAME}"

echo "=== FastSAM3DToOpenSim AWS Batch Job ==="
echo "  Input:  $S3_INPUT_URI"
echo "  Output: $S3_OUTPUT_URI"
echo "  Type:   $INFERENCE_TYPE"

# ---- Pull video from S3 -----------------------------------------------------
echo ">>> Downloading video from S3..."
mkdir -p /tmp/input
aws s3 cp "$S3_INPUT_URI" "$LOCAL_VIDEO"

# ---- Activate conda env & run ------------------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body

echo ">>> Processing video..."
python demo_video_opensim.py \
    --video_path "$LOCAL_VIDEO" \
    --output_dir "$OUTPUT_DIR" \
    --inference_type "$INFERENCE_TYPE" \
    ${EXTRA_ARGS:-}

# ---- Optional: GLB export via Blender ---------------------------------------
# Set EXPORT_GLB=1 to convert the .mot output to a rigged GLB
if [ "${EXPORT_GLB:-0}" = "1" ]; then
    MOT_FILE=$(find "$OUTPUT_DIR" -name "*.mot" | head -n 1)
    TRC_FILE=$(find "$OUTPUT_DIR" -name "*.trc" | head -n 1)
    if [ -n "$MOT_FILE" ]; then
        echo ">>> Exporting GLB via Blender..."
        blender --background --python export_glb_skely.py -- \
            --mot "$MOT_FILE" \
            --output "${OUTPUT_DIR}/${VIDEO_NAME}.glb" \
            ${TRC_FILE:+--trc "$TRC_FILE"}
    fi
fi

# ---- Push results to S3 -----------------------------------------------------
echo ">>> Uploading results to S3..."
aws s3 cp "$OUTPUT_DIR" "${S3_OUTPUT_URI}output_${TIMESTAMP}_${VIDEO_NAME}/" --recursive

echo "=== Job complete ==="
