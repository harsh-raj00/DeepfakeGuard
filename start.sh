#!/bin/bash

# ====================================================================
# FaceAuth Guard - Cloud Production Startup Script
# ====================================================================

echo "[STARTUP] Downloading Machine Learning Models..."
python scripts/download_models.py

echo "[STARTUP] Starting Gunicorn WSGI Server with Eventlet..."
# Render.com provides the PORT environment variable automatically
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:${PORT:-5000} run:app
