"""
=============================================================================
 MesoNet Trainer — Train on Kaggle "Deepfake and Real Images" Dataset
 
 Dataset: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
 Author:  Manjil Karki (Sourced from Zenodo)
 
 This script loads the real Kaggle dataset, applies data augmentation,
 trains the Meso4 CNN, generates training curves, confusion matrix,
 and saves the best weights.

 Usage:
   python scripts/train_mesonet_kaggle.py
   python scripts/train_mesonet_kaggle.py --dataset_path /path/to/Dataset
   python scripts/train_mesonet_kaggle.py --epochs 20 --batch_size 64
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def find_dataset_path(provided_path=None):
    """
    Auto-detect the Kaggle dataset path.
    Searches common locations if no path is provided.

    Expected structure: {root}/Train/Real, {root}/Train/Fake, etc.
    """
    candidates = [
        provided_path,
        config.KAGGLE_DATASET_PATH,
        os.path.join(config.BASE_DIR, "Dataset"),
        os.path.join(config.BASE_DIR, "dataset"),
        os.path.join(config.BASE_DIR, "data", "Dataset"),
        os.path.join(config.BASE_DIR, "deepfake-and-real-images", "Dataset"),
    ]

    for path in candidates:
        if path and os.path.isdir(path):
            # Check for expected structure
            train_dir = os.path.join(path, "Train")
            if not os.path.isdir(train_dir):
                # Maybe one level deeper
                for sub in os.listdir(path):
                    sub_path = os.path.join(path, sub)
                    if os.path.isdir(os.path.join(sub_path, "Train")):
                        return sub_path
            else:
                return path

    return None


def count_images(directory):
    """Count images in a directory recursively."""
    count = 0
    if not os.path.isdir(directory):
        return 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    return count


def train_mesonet_kaggle(dataset_path, epochs=None, batch_size=None, lr=None):
    """
    Train MesoNet on the Kaggle Deepfake and Real Images dataset.

    Args:
        dataset_path (str): Path to dataset root (containing Train/, Validation/, Test/)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )

    # Import the model builder
    from ml.deepfake_detector import build_meso4_model

    # ── Configuration ──
    epochs = epochs or config.TRAINING_EPOCHS
    batch_size = batch_size or config.TRAINING_BATCH_SIZE
    lr = lr or config.TRAINING_LEARNING_RATE
    img_size = config.TRAINING_IMAGE_SIZE

    print("=" * 65)
    print("  MesoNet Trainer — Kaggle Deepfake & Real Images Dataset")
    print("=" * 65)
    print(f"  Dataset:    {dataset_path}")
    print(f"  Image Size: {img_size}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learn Rate: {lr}")
    print("-" * 65)

    # ── Verify dataset structure ──
    train_dir = os.path.join(dataset_path, "Train")
    val_dir = os.path.join(dataset_path, "Validation")
    test_dir = os.path.join(dataset_path, "Test")

    # Some dataset versions use "Valid" instead of "Validation"
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(dataset_path, "Valid")

    dirs_ok = True
    for name, d in [("Train", train_dir), ("Validation", val_dir), ("Test", test_dir)]:
        if os.path.isdir(d):
            real_count = count_images(os.path.join(d, "Real"))
            fake_count = count_images(os.path.join(d, "Fake"))
            print(f"  {name:12s}: {real_count:>6} Real, {fake_count:>6} Fake")
        else:
            print(f"  {name:12s}: [NOT FOUND] at {d}")
            if name == "Validation":
                val_dir = None  # Will use validation_split instead
            else:
                dirs_ok = False

    if not dirs_ok:
        print("\n[ERROR] Required folders (Train, Test) not found!")
        print(f"  Expected structure: {dataset_path}/Train/Real, {dataset_path}/Train/Fake, etc.")
        return False

    print("-" * 65)

    # ── Data Generators with Augmentation ──
    print("\n[INFO] Setting up data generators with augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.15 if val_dir is None else 0.0
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # Load training data
    # Classes: Fake=0, Real=1 (alphabetical order)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        subset='training' if val_dir is None else None
    )

    # Load validation data
    if val_dir and os.path.isdir(val_dir):
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
    else:
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            subset='validation'
        )

    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    print(f"\n  Class mapping: {train_generator.class_indices}")
    print(f"  Training samples:   {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Test samples:       {test_generator.samples}")

    # ── Build Model ──
    print("\n[INFO] Building Meso4 model...")
    model = build_meso4_model(input_shape=(img_size[0], img_size[1], 3))
    if model is None:
        print("[ERROR] Model build failed!")
        return False

    # Recompile with specific learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")

    # ── Callbacks ──
    weights_path = config.DEEPFAKE_MODEL_PATH
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            weights_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ── Train ──
    print("\n" + "=" * 65)
    print("  TRAINING STARTED")
    print("=" * 65)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate on Test Set ──
    print("\n" + "=" * 65)
    print("  EVALUATION ON TEST SET")
    print("=" * 65)

    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"\n  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Test Loss:     {test_loss:.4f}")

    # ── Confusion Matrix & Detailed Metrics ──
    print("\n[INFO] Computing confusion matrix...")
    test_generator.reset()

    y_pred_prob = model.predict(test_generator, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    # Ensure same length
    min_len = min(len(y_pred), len(y_true))
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0

    print("\n  CONFUSION MATRIX")
    print("  " + "-" * 45)
    print(f"                        Predicted")
    print(f"                     Real    |  Fake")
    print(f"   Actual Real    | TP={TP:>5} | FN={FN:>5}")
    print(f"   Actual Fake    | FP={FP:>5} | TN={TN:>5}")
    print("  " + "-" * 45)

    print("\n  METRICS")
    print("  " + "-" * 45)
    print(f"   Accuracy:    {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"   Precision:   {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"   Recall:      {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"   F1-Score:    {f1:.4f}  ({f1 * 100:.2f}%)")
    print(f"   FAR:         {far:.4f}  ({far * 100:.2f}%)")
    print(f"   FRR:         {frr:.4f}  ({frr * 100:.2f}%)")

    # ── Save Training Results ──
    results_dir = config.TRAINING_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics JSON
    metrics = {
        'dataset': 'Kaggle Deepfake and Real Images (Manjil Karki)',
        'dataset_url': 'https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images',
        'training_samples': train_generator.samples,
        'validation_samples': val_generator.samples,
        'test_samples': test_generator.samples,
        'epochs_trained': len(history.history['accuracy']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'far': round(far, 4),
        'frr': round(frr, 4),
        'model_params': total_params,
        'image_size': list(img_size),
        'batch_size': batch_size,
        'initial_lr': lr,
        'timestamp': datetime.now().isoformat(),
        'class_mapping': train_generator.class_indices
    }

    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")

    # ── Save Training Curves ──
    try:
        save_training_curves(history, results_dir)
        print(f"  Training curves saved to: {results_dir}")
    except Exception as e:
        print(f"  [WARNING] Could not save plots: {e}")
        # Save history as CSV fallback
        save_history_csv(history, results_dir)

    # ── Final Summary ──
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best weights saved to: {weights_path}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  F1-Score:      {f1 * 100:.2f}%")
    print("=" * 65)

    return True


def save_training_curves(history, output_dir):
    """Save training/validation accuracy and loss curves as images."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['accuracy']) + 1)

    # ── Accuracy Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy', markersize=4)
    ax1.plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation Accuracy', markersize=4)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.05])

    # ── Loss Plot ──
    ax2.plot(epochs, history.history['loss'], 'b-o', label='Training Loss', markersize=4)
    ax2.plot(epochs, history.history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("  [SAVED] training_curves.png")


def save_history_csv(history, output_dir):
    """Save training history as CSV (fallback if matplotlib not available)."""
    csv_path = os.path.join(output_dir, 'training_history.csv')
    with open(csv_path, 'w') as f:
        keys = list(history.history.keys())
        f.write(','.join(['epoch'] + keys) + '\n')
        for i in range(len(history.history[keys[0]])):
            row = [str(i + 1)] + [f"{history.history[k][i]:.6f}" for k in keys]
            f.write(','.join(row) + '\n')
    print(f"  [SAVED] training_history.csv")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train MesoNet on Kaggle Deepfake & Real Images dataset'
    )
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the dataset root (containing Train/, Test/, Validation/)')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {config.TRAINING_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=None,
                        help=f'Batch size (default: {config.TRAINING_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Learning rate (default: {config.TRAINING_LEARNING_RATE})')

    args = parser.parse_args()

    # Find dataset
    ds_path = find_dataset_path(args.dataset_path)

    if ds_path is None:
        print("=" * 65)
        print("  ERROR: Kaggle dataset not found!")
        print("=" * 65)
        print()
        print("  Please download the dataset from:")
        print("  https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images")
        print()
        print("  Then extract it to one of these locations:")
        print(f"    1. {config.KAGGLE_DATASET_PATH}")
        print(f"    2. {config.BASE_DIR}/Dataset")
        print()
        print("  Or specify the path explicitly:")
        print("    python scripts/train_mesonet_kaggle.py --dataset_path /path/to/Dataset")
        print()
        print("  Expected structure:")
        print("    Dataset/")
        print("    +-- Train/Real/  +  Train/Fake/")
        print("    +-- Validation/Real/  +  Validation/Fake/")
        print("    +-- Test/Real/  +  Test/Fake/")
        print()

        # Offer to train with synthetic data instead
        print("  [FALLBACK] Training with synthetic data instead...")
        print("  Run 'python scripts/train_demo_mesonet.py' for synthetic training.")
        sys.exit(1)
    else:
        success = train_mesonet_kaggle(
            ds_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        if not success:
            sys.exit(1)
