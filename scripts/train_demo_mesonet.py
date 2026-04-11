"""
=============================================================================
 MesoNet Demo Trainer — Generates synthetic training data and trains the
 Meso4 model so deepfake detection produces meaningful confidence scores.

 Real faces:  smooth, natural textures with skin-like color distributions
 Fake faces:  contain GAN-like artifacts — checkerboard, noise, frequency
              anomalies in the high-frequency domain

 Usage:  python scripts/train_demo_mesonet.py
 Output: ml/models/mesonet_weights.h5
=============================================================================
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_synthetic_data(num_samples=500, img_size=256):
    """
    Generate synthetic training data for MesoNet.

    REAL images: smooth Gaussian blobs with skin-tone gradients and
                 natural-looking texture (simulating real face patches).

    FAKE images: same base with injected GAN artifacts:
                 - Checkerboard patterns (common in GAN upsampling)
                 - High-frequency noise (compression artifacts)
                 - Periodic stripe patterns
                 - Color channel inconsistencies

    Args:
        num_samples: number of samples per class
        img_size: image resolution (square)

    Returns:
        X (np.ndarray): images, shape (2*num_samples, img_size, img_size, 3)
        y (np.ndarray): labels, 1=REAL, 0=FAKE
    """
    print(f"[INFO] Generating {num_samples * 2} synthetic training images...")

    images = []
    labels = []

    for i in range(num_samples):
        # ── Generate REAL-looking face patch ──
        real = _generate_real_patch(img_size)
        images.append(real)
        labels.append(1)  # REAL

        # ── Generate FAKE-looking face patch ──
        fake = _generate_fake_patch(img_size)
        images.append(fake)
        labels.append(0)  # FAKE

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.float32)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def _generate_real_patch(size):
    """Generate a smooth, natural-looking face-like patch."""
    # Base: smooth Gaussian noise in skin-tone range
    patch = np.zeros((size, size, 3), dtype=np.float32)

    # Skin-tone base color (randomized)
    base_r = np.random.randint(150, 220)
    base_g = np.random.randint(100, 170)
    base_b = np.random.randint(80, 140)

    patch[:, :, 0] = base_r
    patch[:, :, 1] = base_g
    patch[:, :, 2] = base_b

    # Add smooth gradients (simulating facial contours)
    for _ in range(3):
        cx, cy = np.random.randint(20, size - 20, 2)
        radius = np.random.randint(40, size // 2)
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = np.clip(1.0 - dist / radius, 0, 1)
        intensity = np.random.uniform(-30, 30)
        for c in range(3):
            patch[:, :, c] += mask * intensity

    # Add very subtle natural noise
    noise = np.random.normal(0, 3, (size, size, 3))
    patch += noise

    return np.clip(patch, 0, 255).astype(np.uint8)


def _generate_fake_patch(size):
    """Generate a face patch with GAN-like artifacts."""
    # Start with a real-looking base
    patch = _generate_real_patch(size).astype(np.float32)

    # ── Artifact 1: Checkerboard pattern (GAN upsampling artifact) ──
    checker = np.zeros((size, size), dtype=np.float32)
    block = np.random.choice([2, 4, 8])
    for i in range(0, size, block):
        for j in range(0, size, block):
            if (i // block + j // block) % 2 == 0:
                checker[i:i + block, j:j + block] = 1.0
    intensity = np.random.uniform(5, 20)
    for c in range(3):
        patch[:, :, c] += checker * intensity

    # ── Artifact 2: High-frequency noise bands ──
    freq_noise = np.random.normal(0, np.random.uniform(8, 25), (size, size, 3))
    # Apply in patches
    mask = np.random.random((size, size, 1)) > 0.5
    patch += freq_noise * mask

    # ── Artifact 3: Periodic stripe pattern (GAN grid artifacts) ──
    stripe_freq = np.random.randint(4, 16)
    stripes = np.sin(np.linspace(0, stripe_freq * np.pi, size)).reshape(1, size, 1)
    stripe_intensity = np.random.uniform(3, 12)
    patch += stripes * stripe_intensity

    # ── Artifact 4: Color channel shift (inconsistent color blending) ──
    channel = np.random.randint(0, 3)
    shift = np.random.uniform(5, 15)
    patch[:, :, channel] += shift

    # ── Artifact 5: Local blur inconsistency (blending boundary artifacts) ──
    region_y = np.random.randint(0, size // 2)
    region_x = np.random.randint(0, size // 2)
    region_h = np.random.randint(30, size // 3)
    region_w = np.random.randint(30, size // 3)
    import cv2
    region = patch[region_y:region_y + region_h, region_x:region_x + region_w]
    if region.size > 0:
        blurred = cv2.GaussianBlur(region, (5, 5), 2.0)
        patch[region_y:region_y + region_h, region_x:region_x + region_w] = blurred

    return np.clip(patch, 0, 255).astype(np.uint8)


def train_mesonet():
    """Train the Meso4 model on synthetic data and save weights."""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("[ERROR] TensorFlow not available. Cannot train MesoNet.")
        print("        Install with: pip install tensorflow-cpu")
        return False

    # Import model builder
    from ml.deepfake_detector import build_meso4_model

    print("=" * 60)
    print(" MesoNet Demo Trainer — Synthetic Data")
    print("=" * 60)

    # Generate data
    X, y = generate_synthetic_data(num_samples=600, img_size=256)

    # Split into train/validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Validation set: {len(X_val)} samples")

    # Build model
    model = build_meso4_model(input_shape=(256, 256, 3))
    if model is None:
        print("[ERROR] Model build failed.")
        return False

    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    # Train
    print("\n[INFO] Training MesoNet...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    # Final metrics
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULT] Validation Accuracy: {val_acc:.4f}")
    print(f"[RESULT] Validation Loss:     {val_loss:.4f}")

    # Save weights
    weights_path = config.DEEPFAKE_MODEL_PATH
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)
    print(f"\n[SAVED] Weights saved to: {weights_path}")

    # Also save full model for reference
    model_path = weights_path.replace('.weights.h5', '_full.keras')
    model.save(model_path)
    print(f"[SAVED] Full model saved to: {model_path}")

    print("\n" + "=" * 60)
    print(" MesoNet training complete!")
    print(" The deepfake detector will now produce meaningful results.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = train_mesonet()
    if not success:
        print("\n[FAILED] MesoNet training failed. Check errors above.")
        sys.exit(1)
