# ── Base image: Python 3.12 slim ──────────────────────────────────────────────
FROM python:3.12-slim

# System dependencies for audio processing / librosa / webrtcvad
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch ───────────────────────────────────────────────────────────
# Default: CPU. Set BUILD_TARGET=gpu in docker-compose to use CUDA 12.1
ARG BUILD_TARGET=cpu

RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$BUILD_TARGET" = "gpu" ]; then \
        pip install --no-cache-dir \
            torch \
            torchaudio \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir \
            torch \
            torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── Install remaining requirements ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#docker compose --profile gpu up --build