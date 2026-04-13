#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body

echo "=== FastSAM3DToOpenSim Docker Entrypoint ==="

CMD_ARGS=("$@")

# TensorRT engine generation — engines are GPU-specific and must be rebuilt
# when switching GPU type. Set GENERATE_TRT=1 (default) to auto-build missing
# engines, FORCE_GENERATE_TRT=1 to rebuild all.
GENERATE_TRT=${GENERATE_TRT:-1}
FORCE_GENERATE_TRT=${FORCE_GENERATE_TRT:-0}

if [ "$GENERATE_TRT" = "1" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
    echo "--- Checking/generating TensorRT engines ---"

    # YOLO pose detector
    if [ ! -f "checkpoints/yolo/yolo11m-pose.engine" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
        echo ">>> Building YOLO engine..."
        python -c "from ultralytics import YOLO; model = YOLO('checkpoints/yolo/yolo11m-pose.pt'); model.export(format='engine', device=0, half=True, imgsz=640)"
    fi

    # MoGe depth estimator
    if [ ! -f "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
        echo ">>> Building MoGe engine..."
        python convert_moge_encoder_trt.py --all
    fi

    # DINOv3 backbone (USE_TRT_BACKBONE=0 during export to avoid AttributeError)
    if [ ! -f "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine" ] || [ "$FORCE_GENERATE_TRT" = "1" ]; then
        echo ">>> Building Backbone engine (this may take a while)..."
        USE_TRT_BACKBONE=0 python convert_backbone_tensorrt.py --all
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
