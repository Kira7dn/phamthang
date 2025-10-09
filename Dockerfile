# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system libs for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.prod.txt

# Cleanup unnecessary cache and metadata
RUN rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.12 -name '__pycache__' -exec rm -rf {} + && \
    find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete


# ============================================
# Stage 2: Final image
# ============================================
FROM python:3.12-slim

WORKDIR /app

# Copy only installed dependencies from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY . .

# Create necessary directories and permissions
RUN mkdir -p /app/outputs /app/tmp \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
