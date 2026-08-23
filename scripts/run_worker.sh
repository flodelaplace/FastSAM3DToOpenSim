#!/bin/bash
# Worker persistant : prépare l'environnement UNE fois, puis consomme SQS.
#
# Variables d'environnement :
#   WORK_QUEUE_URL         file SQS des vidéos à traiter          (obligatoire)
#   CHECKPOINTS_S3_URI     s3://…/checkpoints/                    (obligatoire)
#   SNS_TOPIC_ARN          notifications de fin                   (facultatif)
#   IDLE_EXIT_SECONDS      extinction après N s sans message      (défaut 600)
#   MAX_JOBS_PER_WORKER    redémarrage préventif                  (défaut 20)
#
# À comparer avec scripts/run_job.sh, qui traite UNE vidéo puis meurt : ici la
# préparation ci-dessous (sync des checkpoints, activation conda) est payée une
# fois pour toutes les vidéos de la session, au lieu d'une fois par vidéo.
#
# DETTE ASSUMÉE : le bloc de sync est copié de run_job.sh plutôt que factorisé.
# C'est délibéré — run_job.sh fait tourner la production, et on ne déstabilise
# pas un chemin qui marche pour éviter une duplication dans un chemin qui n'a
# pas encore fait ses preuves. À factoriser dans scripts/_prepare_env.sh dès que
# le worker aura remplacé run_job.sh sur la voie interactive.
set -euo pipefail

: "${WORK_QUEUE_URL:?WORK_QUEUE_URL is not set}"
: "${CHECKPOINTS_S3_URI:?CHECKPOINTS_S3_URI is not set}"

echo "=== FastSAM3DToOpenSim — worker persistant ==="
echo "  File SQS     : $WORK_QUEUE_URL"
echo "  Checkpoints  : $CHECKPOINTS_S3_URI"
echo "  Inactivité   : ${IDLE_EXIT_SECONDS:-600} s avant extinction"

# ---- Sync checkpoints depuis S3 ---------------------------------------------
# Identique à run_job.sh : s5cmd si disponible (~256 flux parallèles, 5-10× plus
# rapide sur le bundle DINOv3 et ses ~498 petits fichiers), repli automatique
# sur aws s3 sync. Les weights ONNX externes sont exclus : seul le .engine est
# chargé au runtime.
echo ">>> Sync des checkpoints..."
mkdir -p /app/checkpoints
SYNC_OK=0
SYNC_START=$(date +%s)
if command -v s5cmd >/dev/null 2>&1; then
    if s5cmd sync --exclude "*encoder.*" --exclude "*onnx__*" \
        "${CHECKPOINTS_S3_URI%/}/*" /app/checkpoints/; then
        SYNC_OK=1
        echo ">>> Sync OK via s5cmd ($(( $(date +%s) - SYNC_START )) s)"
    else
        echo ">>> WARN: s5cmd a échoué (exit $?), repli sur aws s3 sync"
    fi
fi
if [ "$SYNC_OK" = "0" ]; then
    aws s3 sync "$CHECKPOINTS_S3_URI" /app/checkpoints/ \
        --exclude "*encoder.*" --exclude "*onnx__*" --no-progress
    echo ">>> Sync OK via aws s3 sync ($(( $(date +%s) - SYNC_START )) s)"
fi

# ---- Environnement Python ----------------------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body
cd /app

# ---- Boucle -----------------------------------------------------------------
# exec : le worker devient PID 1, donc SIGTERM (reprise Spot, arrêt Batch) lui
# parvient directement au lieu d'être avalé par ce script.
echo ">>> Démarrage de la boucle"
exec python -m worker.loop
