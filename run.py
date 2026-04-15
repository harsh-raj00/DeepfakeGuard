"""
=============================================================================
 Application Entry Point — Face Authentication System
 Run this to start the Flask + SocketIO server.

 Usage:  python run.py
 Access: http://localhost:5000
=============================================================================
"""

import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from app import create_app, socketio

# ── Create the Flask application ───────────────────────────────────────────
app = create_app()


if __name__ == '__main__':
    # Set console encoding for Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print(" FaceAuth Guard - Anti-Deepfake Authentication System")
    print("=" * 60)
    print(f" Server:   http://localhost:5000")
    print(f" Debug:    {config.DEBUG}")
    print(f" Database: {config.DATABASE_PATH}")
    print(f" Models:   {config.MODELS_DIR}")
    print("=" * 60)

    # ── Check if models are downloaded ──
    models_ok = True
    required = [
        ("Face Detection (prototxt)", config.FACE_DETECTION_MODEL_PROTO),
        ("Face Detection (weights)", config.FACE_DETECTION_MODEL_WEIGHTS),
    ]

    for name, path in required:
        if os.path.exists(path):
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name}")
            models_ok = False

    optional = [
        ("Shape Predictor (dlib)", config.SHAPE_PREDICTOR_PATH),
        ("MesoNet Weights", config.DEEPFAKE_MODEL_PATH),
    ]

    for name, path in optional:
        if os.path.exists(path):
            print(f"  [OK] {name}")
        else:
            print(f"  [FALLBACK] {name} - not found (will use fallback)")

    if not models_ok:
        print("\n[WARNING] Some required models are missing!")
        print("   Run: python scripts/download_models.py")
        print("   Or the system will try to start in limited mode.\n")

    print("=" * 60)
    print("Pre-loading all ML models in main thread... This may take a moment.")
    from app.socketio_events import get_ml_modules
    get_ml_modules()
    print("ML Models pre-loaded successfully!")
    print("=" * 60)

    # ── Start the server ──
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=config.DEBUG,
        allow_unsafe_werkzeug=True
    )
