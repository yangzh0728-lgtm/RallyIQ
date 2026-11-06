# RallyIQ analysis worker (NVIDIA GPU). Pulls jobs from the shared PostgreSQL queue, reads/writes
# videos and artifacts in object storage. Model weights are baked into the image at build time.
#
#   docker build -f docker/worker.Dockerfile -t rallyiq-worker .
#   docker run --gpus all --env-file .env rallyiq-worker                      # long-running
#   docker run --gpus all --env-file .env rallyiq-worker --idle-exit 300      # on-demand: exit when idle
#
# Build args: POSE_VARIANT=rtmo-s|rtmo-m, FETCH_WEIGHTS=0 to skip downloads (worker then emits empty
# tracks for those stages, useful for smoke tests without network).

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG POSE_VARIANT=rtmo-s
ARG FETCH_WEIGHTS=1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive \
    RALLYIQ_DATA=/tmp/rallyiq \
    RALLYIQ_POSE_MODEL=/opt/rallyiq/weights/pose.onnx \
    RALLYIQ_SHUTTLE_WEIGHTS=/opt/rallyiq/weights/shuttle/heatmap.pt
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY rallyiq/ rallyiq/
COPY scripts/ scripts/
# torch already comes with the base image; onnxruntime-gpu runs the pose model on CUDA.
RUN pip install '.[server,shuttle]' opencv-python-headless onnxruntime-gpu \
    && pip uninstall -y opencv-python || true
RUN if [ "$FETCH_WEIGHTS" = 1 ]; then WEIGHTS_DIR=/opt/rallyiq/weights POSE_VARIANT=$POSE_VARIANT bash scripts/fetch_weights.sh; fi
ENTRYPOINT ["python", "-m", "rallyiq.pipeline", "worker"]
