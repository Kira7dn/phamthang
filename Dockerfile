# ==========================================
# Stage 1: Build dependencies
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /install

# Cài build deps cho OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc pkg-config libjpeg-dev zlib1g-dev libpng-dev \
    libtiff-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.prod.txt .

# Install to /install prefix to copy only site-packages
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -r requirements.prod.txt \
    && rm -rf /root/.cache/pip \
    && find /install -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type f -name '*.pyc' -delete \
    && find /install -type f -name '*.pyo' -delete


# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# Install only runtime libs (minimal set for opencv-headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libtiff6 libfreetype6 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only Python packages from builder (not entire /usr/local)
COPY --from=builder /install /usr/local

# Copy entrypoint first (smaller layer)
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Create user, directories, and set permissions in one layer
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/outputs /app/tmp \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application code (do this last to leverage cache)
# PYTHONDONTWRITEBYTECODE=1 prevents .pyc creation, .dockerignore excludes __pycache__
COPY --chown=appuser:appuser . .

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
