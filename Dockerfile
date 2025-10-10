# ==========================================
# Stage 1: Build dependencies
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Cài build deps cho OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc pkg-config libjpeg-dev zlib1g-dev libpng-dev \
    libtiff-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.prod.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.prod.txt \
    && rm -rf /root/.cache/pip


# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libtiff6 libfreetype6 \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Sao chép dependency từ builder
COPY --from=builder /usr/local /usr/local

# Sao chép code ứng dụng
COPY . .

# Tạo user runtime & thư mục
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/outputs /app/tmp \
    && chown -R appuser:appuser /app

# Thiết lập entrypoint đảm bảo quyền ghi trước khi vào app user
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
