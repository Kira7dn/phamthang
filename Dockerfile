# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (for opencv, pillow, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc pkg-config libjpeg-dev zlib1g-dev libpng-dev \
    libtiff-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.base.txt .

# Install Python packages into /install (prefix for clean copy later)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.base.txt && \
    rm -rf /root/.cache/pip && \
    find / -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Copy entire project
COPY . .

# ==========================================
# Stage 2: Runtime (Local / VPS)
# ==========================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime libraries needed by OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libtiff6 libfreetype6 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy Python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=1000:1000 . .

# Copy and prepare entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user (safety)
RUN useradd -m -u 1000 appuser || true

EXPOSE 8000

# Use our entrypoint
ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
