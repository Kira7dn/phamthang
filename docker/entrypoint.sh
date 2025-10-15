#!/bin/sh
set -e

echo "[ENTRYPOINT] Preparing runtime directories..."
mkdir -p /app/outputs /app/tmp 2>/dev/null || true

echo "[ENTRYPOINT] Starting application as $(id -u):$(id -g)"
exec "$@"
