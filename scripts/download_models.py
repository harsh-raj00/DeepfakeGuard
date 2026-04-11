"""
=============================================================================
 Model Downloader — Automatically downloads pretrained model weights.
 Run this script before first use:  python scripts/download_models.py
=============================================================================
"""

import os
import sys
import requests
import bz2
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_file(url, dest_path, description="file"):
    """
    Download a file from URL with progress display.

    Args:
        url (str): Download URL.
        dest_path (str): Local file path to save to.
        description (str): Human-readable name for progress display.
    """
    if os.path.exists(dest_path):
        print(f"  [SKIP] {description} already exists at {dest_path}")
        return True

    print(f"  [DOWNLOADING] {description}...")
    print(f"    URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r    Progress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='')

        print(f"\n  [DONE] Saved to {dest_path}")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Failed to download {description}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_all_models():
    """Download all required model files."""
    print("=" * 60)
    print(" Face Authentication System — Model Downloader")
    print("=" * 60)

    models_dir = config.MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    success_count = 0
    total_count = len(config.MODEL_URLS)

    # ── Download each model ──
    for filename, url in config.MODEL_URLS.items():
        dest_path = os.path.join(models_dir, filename)
        print(f"\n[{success_count + 1}/{total_count}] {filename}")

        if download_file(url, dest_path, filename):
            success_count += 1

    # ── Handle shape_predictor bz2 extraction ──
    predictor_path = config.SHAPE_PREDICTOR_PATH
    predictor_bz2 = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat.bz2")

    if not os.path.exists(predictor_path) and os.path.exists(predictor_bz2):
        print(f"\n[EXTRACTING] shape_predictor_68_face_landmarks.dat from .bz2...")
        try:
            with bz2.BZ2File(predictor_bz2) as fr, open(predictor_path, 'wb') as fw:
                shutil.copyfileobj(fr, fw)
            print(f"  [DONE] Extracted to {predictor_path}")
            os.remove(predictor_bz2)
        except Exception as e:
            print(f"  [ERROR] Extraction failed: {e}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f" Download Summary: {success_count}/{total_count} models ready")
    print("=" * 60)

    # Check all required files
    required_files = {
        "Face Detection (prototxt)": config.FACE_DETECTION_MODEL_PROTO,
        "Face Detection (caffemodel)": config.FACE_DETECTION_MODEL_WEIGHTS,
        "Shape Predictor (68 landmarks)": config.SHAPE_PREDICTOR_PATH,
    }

    all_ready = True
    for name, path in required_files.items():
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        if not os.path.exists(path):
            all_ready = False
        print(f"  {status} {name}")

    # MesoNet weights (optional - model works in demo mode without them)
    meso_status = "[OK]" if os.path.exists(config.DEEPFAKE_MODEL_PATH) else "[DEMO MODE]"
    print(f"  {meso_status} MesoNet Weights")

    if all_ready:
        print("\n[OK] All core models are ready! You can now run the application.")
    else:
        print("\n[WARNING] Some models are missing. The system may run in limited mode.")

    return all_ready


if __name__ == "__main__":
    download_all_models()
