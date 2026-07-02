#!/bin/bash
# =============================================================================
# run_job.sh — AWS Batch job wrapper
#
# Expected environment variables:
#   S3_INPUT_URI         s3://bucket/input/video.mp4            (required)
#   S3_OUTPUT_URI        s3://bucket/output/                     (required, trailing /)
#   CHECKPOINTS_S3_URI   s3://bucket/checkpoints/                (required, trailing /)
#   MODE                 sam3d | avatar                          (default: sam3d)
#   INFERENCE_TYPE       body | full                             (default: body)
#   EXTRA_ARGS           per-video flags from filename parser    (default: empty)
#   TRIM_START           integer seconds (ffmpeg -ss)            (optional)
#   TRIM_END             integer seconds (ffmpeg -to)            (optional)
#
# Flow: sync checkpoints → pull video → [ffmpeg trim] → process → push to S3
#
# MODE switches the Python entry point (same Docker image, same EXTRA_ARGS grammar):
#   MODE=sam3d   → demo_video_opensim.py  (full pipeline: TRC + IK + mesh + anat GLB + .mot)
#                  Output S3 layout:   <S3_OUTPUT_URI>/output_<TS>_<name>/{trc,mot,glb,...}
#   MODE=avatar  → generate_avatars.py    (skip OpenSim, TRC + N humanised avatar GLBs only)
#                  Output S3 layout:   <S3_OUTPUT_URI>/{template_id}.glb        (flat under
#                  the prefix — the Lambda already builds it as
#                  02-output-avatar/<user>/<exercise>/)
# =============================================================================
set -euo pipefail

# ---- Validate required vars -------------------------------------------------
: "${S3_INPUT_URI:?S3_INPUT_URI is not set}"
: "${S3_OUTPUT_URI:?S3_OUTPUT_URI is not set}"
: "${CHECKPOINTS_S3_URI:?CHECKPOINTS_S3_URI is not set}"

MODE="${MODE:-sam3d}"
INFERENCE_TYPE="${INFERENCE_TYPE:-body}"
VIDEO_BASENAME=$(basename "$S3_INPUT_URI")
VIDEO_NAME="${VIDEO_BASENAME%.*}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_INPUT="/tmp/input/${VIDEO_BASENAME}"
PROC_INPUT="$LOCAL_INPUT"

# In avatar mode we write outputs DIRECTLY under S3_OUTPUT_URI (no extra
# `output_<TS>_<name>/` wrapper) — the Lambda has already crafted the prefix
# as 02-output-avatar/<user>/<exercise>/. In sam3d mode we keep the timestamped
# wrapper since the user/exercise grouping doesn't exist there.
if [ "$MODE" = "avatar" ]; then
    OUTPUT_DIR="/outputs/avatar_${TIMESTAMP}_${VIDEO_NAME}"
    S3_OUTPUT_PATH="${S3_OUTPUT_URI%/}/"
else
    OUTPUT_DIR="/outputs/output_${TIMESTAMP}_${VIDEO_NAME}"
    S3_OUTPUT_PATH="${S3_OUTPUT_URI%/}/output_${TIMESTAMP}_${VIDEO_NAME}/"
fi

echo "=== FastSAM3DToOpenSim AWS Batch Job ==="
echo "  Mode:          $MODE"
echo "  Input:         $S3_INPUT_URI"
echo "  Output S3:     $S3_OUTPUT_PATH"
echo "  Checkpoints:   $CHECKPOINTS_S3_URI"
echo "  Inference:     $INFERENCE_TYPE"
echo "  Extra args:    ${EXTRA_ARGS:-<none>}"
echo "  Trim:          ${TRIM_START:-0}s -> ${TRIM_END:-<end>}"

# ---- Sync checkpoints from S3 -----------------------------------------------
# Chemin rapide : s5cmd (~256 streams parallèles, 5-10× plus rapide qu'aws
# s3 sync sur le bundle DINOv3 ~498 petits fichiers). Si s5cmd absent ou
# échoue (auth, glob, S3 hiccup), fallback automatique sur `aws s3 sync` —
# aucun chemin de régression possible : ce qui marchait avant marche encore.
echo ">>> Syncing checkpoints from S3..."
mkdir -p /app/checkpoints
SYNC_OK=0
SYNC_START=$(date +%s)
# Exclure les ~500 fichiers de weights ONNX externes du backbone DINOv3 :
# inutiles au runtime (seul le .engine est chargé), et le pré-sync
# entrypoint a déjà pris .engine + .build_skipped. Si un rebuild devient
# nécessaire, l'entrypoint re-sync full le dossier juste avant.
# Nomenclature : les weights sont préfixés "encoder." ou "onnx__" — s5cmd
# match par basename (pas par path), donc ces patterns filtrent bien les
# ~500 fichiers weights sans toucher .onnx / .engine / .build_skipped.
if command -v s5cmd >/dev/null 2>&1; then
    if s5cmd sync --exclude "*encoder.*" --exclude "*onnx__*" \
        "${CHECKPOINTS_S3_URI%/}/*" /app/checkpoints/; then
        SYNC_OK=1
        echo ">>> Sync checkpoints OK via s5cmd ($(( $(date +%s) - SYNC_START )) s)"
    else
        echo ">>> WARN: s5cmd sync a échoué (exit $?), fallback aws s3 sync"
    fi
fi
if [ "$SYNC_OK" = "0" ]; then
    aws s3 sync "$CHECKPOINTS_S3_URI" /app/checkpoints/ \
        --exclude "*encoder.*" --exclude "*onnx__*" --no-progress
    echo ">>> Sync checkpoints OK via aws s3 sync ($(( $(date +%s) - SYNC_START )) s)"
fi

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
# from the filename meta block: h<cm>, s/e trim, st, com, multi-person, floor,
# floor_seated, feet_anchor, etc.).
echo ">>> Processing video (mode=$MODE)..."
if [ "$MODE" = "avatar" ]; then
    # Avatar pipeline : same SAM3D body inference + retarget onto each
    # MakeHuman template in assets/avatars/avatar_*_apose_opaque.glb. Writes
    # markers_<name>.trc + markers_<name>_avatar_<template>.glb per template.
    # shellcheck disable=SC2086
    python generate_avatars.py \
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
else
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
fi

# ---- Push results to S3 -----------------------------------------------------
echo ">>> Uploading results to S3..."
if [ "$MODE" = "avatar" ]; then
    # Avatar : on n'envoie QUE les GLB avatars (pas le TRC, pas l'inference_meta).
    # L'app kiné a juste besoin des avatars finaux.
    AVATAR_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_avatar_*.glb" | wc -l)
    echo ">>> Uploading $AVATAR_COUNT avatar GLB(s) to $S3_OUTPUT_PATH"
    for glb in "$OUTPUT_DIR"/*_avatar_*.glb; do
        [ -f "$glb" ] || continue
        # Strip the markers_<name>_avatar_ prefix → keep only "<template_id>.glb"
        # ex: markers_squat_001__h180_floor_avatar_female_young.glb → female_young.glb
        BASENAME=$(basename "$glb")
        TEMPLATE_ID=$(echo "$BASENAME" | sed -E 's/.*_avatar_(.+)\.glb$/\1.glb/')
        aws s3 cp "$glb" "${S3_OUTPUT_PATH}${TEMPLATE_ID}" --no-progress
    done
else
    aws s3 cp "$OUTPUT_DIR" "$S3_OUTPUT_PATH" --recursive --no-progress
fi

# ---- Cache torch compile artifacts back to S3 ------------------------------
# Persiste les caches compilés PyTorch pour les cold starts suivants
# (Inductor FX graph + Triton CUDA kernels). L'entrypoint pré-sync au start.
# `|| true` : upload best-effort, ne doit pas faire échouer le job.
if [ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ] && [ -d "$TORCHINDUCTOR_CACHE_DIR" ] && \
   [ -n "$(ls -A "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null)" ]; then
    echo ">>> Caching torch inductor artifacts to S3..."
    aws s3 sync "$TORCHINDUCTOR_CACHE_DIR/" \
        "${CHECKPOINTS_S3_URI%/}/torch_inductor_cache/" --no-progress || true
fi
if [ -n "${TRITON_CACHE_DIR:-}" ] && [ -d "$TRITON_CACHE_DIR" ] && \
   [ -n "$(ls -A "$TRITON_CACHE_DIR" 2>/dev/null)" ]; then
    echo ">>> Caching triton kernels to S3..."
    aws s3 sync "$TRITON_CACHE_DIR/" \
        "${CHECKPOINTS_S3_URI%/}/triton_cache/" --no-progress || true
fi

echo "=== Job complete -> $S3_OUTPUT_PATH ==="
