#!/bin/sh
set -e

APP_USER="appuser"
APP_GROUP="appuser"

echo "[ENTRYPOINT] Preparing runtime directories..."
mkdir -p /app/outputs /app/tmp

if [ "$(id -u)" = "0" ]; then
  chown -R "${APP_USER}:${APP_GROUP}" /app/outputs /app/tmp 2>/dev/null || true
  echo "[ENTRYPOINT] Ownership set to ${APP_USER}:${APP_GROUP}"
  echo "[ENTRYPOINT] Launching application as ${APP_USER}:${APP_GROUP}"
  exec gosu "${APP_USER}:${APP_GROUP}" "$@"
fi

echo "[ENTRYPOINT] Already running as $(id -u):$(id -g), starting application"
exec "$@"
