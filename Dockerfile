# RallyIQ web + API (CPU). One container serves the built web UI and /api on port 8000.
# Analysis runs elsewhere: docker/worker.Dockerfile (GPU). See docs/deploy.md.
#
#   docker build -t rallyiq-api .
#   docker run -p 8000:8000 -e RALLYIQ_DEMO=1 rallyiq-api          # local, SQLite + files in /data

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    RALLYIQ_DATA=/data RALLYIQ_WEB_DIR=/app/web RALLYIQ_KNOWLEDGE_DIR=/app/knowledge/coaching
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rallyiq && mkdir -p /data && chown rallyiq /data
WORKDIR /app
COPY pyproject.toml ./
COPY rallyiq/ rallyiq/
# coaching notes the AI coach cites (knowledge/coaching/notes)
COPY knowledge/ knowledge/
# coach: AI coach via Claude API (set ANTHROPIC_API_KEY); headless OpenCV: picture stats for /quality.
RUN pip install '.[server,coach]' opencv-python-headless
COPY --from=web /web/dist /app/web
USER rallyiq
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
# --proxy-headers: trust X-Forwarded-* from the platform's load balancer (client IP for rate limits).
CMD ["sh", "-c", "exec uvicorn rallyiq.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
