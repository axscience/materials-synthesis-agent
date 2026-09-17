# Discovery Harness API -- the Python backend a Lovable/Supabase front end connects to.
# The front end is published on Lovable; this image is the compute backend it calls over HTTPS.
#
#   docker build -t discovery-harness .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... discovery-harness
#
# On Fly/Railway/Render: PORT is injected and the server binds 0.0.0.0 automatically (serve.py).
FROM python:3.12-slim

# Faster, smaller: no .pyc, unbuffered logs, no pip cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tooling some scientific wheels need at install time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# CPU-only torch keeps the image ~2GB smaller than the default CUDA wheel; the harness never uses a GPU.
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[web]"

# Campaign databases live here; mount a volume in production so tenant data survives redeploys.
ENV HARNESS_WORKSPACE=/data
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000
CMD ["python", "-m", "materials_synthesis_agent.harness.serve"]
