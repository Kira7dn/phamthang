#!/bin/sh
# Lambda entrypoint wrapper for FastAPI with Lambda Web Adapter

# Start uvicorn in background
exec uvicorn app.main:app --host 0.0.0.0 --port 8080
