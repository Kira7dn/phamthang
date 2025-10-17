#!/bin/sh
set -e

echo "[ENTRYPOINT] Preparing runtime directories..."

mkdir -p /app/outputs /app/tmp 2>/dev/null || true
chown -R $(id -u):$(id -g) /app/outputs /app/tmp 2>/dev/null || true

echo "[ENTRYPOINT] Starting application as UID:GID -> $(id -u):$(id -g)"
echo "Command: $@"

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
