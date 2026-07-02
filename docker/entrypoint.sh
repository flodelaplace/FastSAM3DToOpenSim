#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body

echo "=== FastSAM3DToOpenSim Docker Entrypoint ==="

CMD_ARGS=("$@")

# Torch compile caches — persist artifacts between cold starts pour éviter
# de recompiler à chaque run (gain ~30-60s sur T4). Deux caches à persister :
#   - TORCHINDUCTOR_CACHE_DIR : FX graph + Inductor codegen artifacts
#   - TRITON_CACHE_DIR        : compiled CUDA kernels (le plus gros gain)
# Le TORCHINDUCTOR_FX_GRAPH_CACHE=1 active explicitement le cache FX (défaut
# on récent PyTorch mais safe). Caches GPU-spécifiques, invalidation auto.
export TORCHINDUCTOR_CACHE_DIR=/app/torch_inductor_cache
export TRITON_CACHE_DIR=/app/triton_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [ -n "$CHECKPOINTS_S3_URI" ]; then
    echo ">>> Pre-sync torch compile caches from S3..."
    aws s3 sync "${CHECKPOINTS_S3_URI%/}/torch_inductor_cache/" "$TORCHINDUCTOR_CACHE_DIR/" \
        --no-progress 2>/dev/null || true
    aws s3 sync "${CHECKPOINTS_S3_URI%/}/triton_cache/" "$TRITON_CACHE_DIR/" \
        --no-progress 2>/dev/null || true
fi

# TensorRT engine generation — engines are GPU-specific and must be rebuilt
# when switching GPU type. Set GENERATE_TRT=1 (default) to auto-build missing
# engines, FORCE_GENERATE_TRT=1 to rebuild all.
GENERATE_TRT=${GENERATE_TRT:-1}
FORCE_GENERATE_TRT=${FORCE_GENERATE_TRT:-0}

if [ "$GENERATE_TRT" = "1" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
    echo "--- Checking/generating TensorRT engines ---"

    # Pré-sync engines depuis S3 pour éviter les rebuilds inutiles au cold
    # start (les engines TRT sont cachés compute-cap-spécifiques, un build
    # YOLO+MoGe coûte ~6 min sur T4). Skippé si FORCE_GENERATE_TRT=1.
    if [ -n "$CHECKPOINTS_S3_URI" ] && [ "$FORCE_GENERATE_TRT" != "1" ]; then
        echo ">>> Pre-sync TRT engines from S3..."
        mkdir -p checkpoints/yolo checkpoints/moge_trt checkpoints/sam-3d-body-dinov3/backbone_trt
        aws s3 sync "${CHECKPOINTS_S3_URI%/}/yolo/" checkpoints/yolo/ \
            --exclude "*" --include "*.engine" --no-progress 2>/dev/null || true
        aws s3 sync "${CHECKPOINTS_S3_URI%/}/moge_trt/" checkpoints/moge_trt/ \
            --exclude "*" --include "*.engine" --no-progress 2>/dev/null || true
        aws s3 sync "${CHECKPOINTS_S3_URI%/}/sam-3d-body-dinov3/backbone_trt/" \
            checkpoints/sam-3d-body-dinov3/backbone_trt/ \
            --exclude "*" --include "*.engine" --include ".build_skipped" \
            --no-progress 2>/dev/null || true
    fi

    # YOLO pose detector
    if [ ! -f "checkpoints/yolo/yolo11m-pose.engine" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
        echo ">>> Building YOLO engine..."
        python -c "from ultralytics import YOLO; model = YOLO('checkpoints/yolo/yolo11m-pose.pt'); model.export(format='engine', device=0, half=True, imgsz=640)"
        if [ -n "$CHECKPOINTS_S3_URI" ] && [ -f "checkpoints/yolo/yolo11m-pose.engine" ]; then
            echo ">>> Caching YOLO engine to S3..."
            aws s3 cp checkpoints/yolo/yolo11m-pose.engine \
                "${CHECKPOINTS_S3_URI%/}/yolo/yolo11m-pose.engine" --no-progress
        fi
    fi

    # MoGe depth estimator
    if [ ! -f "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
        echo ">>> Building MoGe engine..."
        python convert_moge_encoder_trt.py --all
        if [ -n "$CHECKPOINTS_S3_URI" ] && [ -f "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine" ]; then
            echo ">>> Caching MoGe engine to S3..."
            aws s3 cp checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine \
                "${CHECKPOINTS_S3_URI%/}/moge_trt/moge_dinov2_encoder_fp16.engine" --no-progress
        fi
    fi

    # DINOv3 backbone (USE_TRT_BACKBONE=0 during export to avoid AttributeError).
    # On tolère l'échec : sur les GPUs < 16 GB (typiquement RTX 3500 / 3080
    # laptop avec 8-12 GB VRAM), le build TRT peut OOM. Si le build échoue,
    # on crée un marker .build_skipped pour éviter de retenter à chaque run
    # (chaque tentative prend ~5-10 min d'ONNX export avant d'OOM). Le
    # runtime bascule automatiquement sur PyTorch quand le .engine est
    # absent. Pour forcer un retry (ex. après upgrade GPU) : FORCE_GENERATE_TRT=1.
    BACKBONE_ENGINE="checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine"
    BACKBONE_SKIP_MARKER="checkpoints/sam-3d-body-dinov3/backbone_trt/.build_skipped"
    NEED_BACKBONE_BUILD=1
    [ -f "$BACKBONE_ENGINE" ] && NEED_BACKBONE_BUILD=0
    [ -f "$BACKBONE_SKIP_MARKER" ] && [ "$FORCE_GENERATE_TRT" != "1" ] && NEED_BACKBONE_BUILD=0
    if [ "$FORCE_GENERATE_TRT" = "1" ]; then
        NEED_BACKBONE_BUILD=1
        rm -f "$BACKBONE_SKIP_MARKER"
    fi
    if [ "$NEED_BACKBONE_BUILD" = "1" ]; then
        # Le .onnx backbone (700 KB) référence ses weights via external-data
        # dans ~500 fichiers séparés. Le sync global de run_job.sh les exclut
        # (inutiles au runtime) — mais on en a besoin ICI pour build TRT.
        if [ -n "$CHECKPOINTS_S3_URI" ]; then
            echo ">>> Fetching backbone ONNX external weights from S3 for rebuild..."
            aws s3 sync "${CHECKPOINTS_S3_URI%/}/sam-3d-body-dinov3/backbone_trt/" \
                checkpoints/sam-3d-body-dinov3/backbone_trt/ --no-progress 2>/dev/null || true
        fi
        echo ">>> Building Backbone engine (this may take a while)..."
        if USE_TRT_BACKBONE=0 python convert_backbone_tensorrt.py --all; then
            echo ">>> Backbone TRT build success"
            rm -f "$BACKBONE_SKIP_MARKER"
            if [ -n "$CHECKPOINTS_S3_URI" ] && [ -f "$BACKBONE_ENGINE" ]; then
                echo ">>> Caching Backbone engine to S3..."
                aws s3 sync checkpoints/sam-3d-body-dinov3/backbone_trt/ \
                    "${CHECKPOINTS_S3_URI%/}/sam-3d-body-dinov3/backbone_trt/" --no-progress
            fi
        else
            echo ">>> WARNING: Backbone TRT build failed (probably OOM on low-VRAM GPU)."
            echo ">>> Creating skip marker → pas de retry aux runs suivants."
            echo ">>> Runtime bascule sur PyTorch fallback (plus lent mais fonctionnel)."
            mkdir -p "$(dirname "$BACKBONE_SKIP_MARKER")"
            touch "$BACKBONE_SKIP_MARKER"
        fi
    elif [ -f "$BACKBONE_SKIP_MARKER" ] && [ ! -f "$BACKBONE_ENGINE" ]; then
        echo ">>> Backbone TRT build précédemment échoué (marker présent), skip. "
        echo ">>> PyTorch fallback sera utilisé. FORCE_GENERATE_TRT=1 pour retenter."
    fi
fi

# Execute command or fall back to interactive shell
if [ ${#CMD_ARGS[@]} -eq 0 ]; then
    echo "No command provided. Starting interactive shell."
    echo "Example: python demo_video_opensim.py --video_path ./videos/my_video.mp4 --output_dir /outputs/my_run --inference_type body"
    exec bash
else
    echo "Running: ${CMD_ARGS[*]}"
    exec "${CMD_ARGS[@]}"
fi
