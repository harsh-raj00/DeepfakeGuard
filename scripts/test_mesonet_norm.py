import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
from ml.deepfake_detector import DeepfakeDetector

def test_preprocessing():
    d = DeepfakeDetector()
    if not d.model:
        print("Model not loaded, cannot test MesoNet.")
        return

    # Create a simple "real-like" image (some noise and structure)
    img = np.random.randint(100, 150, (256, 256, 3), dtype=np.uint8)
    
    # Preprocess [0, 1]
    p_01 = d.preprocess_face(img)
    
    # Preprocess [-1, 1]
    p_m11 = (img.astype(np.float32) / 127.5) - 1.0
    p_m11 = np.expand_dims(p_m11, axis=0)

    # Preprocess [0, 255] (no norm)
    p_255 = np.expand_dims(img.astype(np.float32), axis=0)

    pred_01 = d.model.predict(p_01, verbose=0)[0][0]
    pred_m11 = d.model.predict(p_m11, verbose=0)[0][0]
    pred_255 = d.model.predict(p_255, verbose=0)[0][0]

    print(f"Predictions for 'Real-like' image:")
    print(f"  [0, 1] norm:   {pred_01:.6f}")
    print(f"  [-1, 1] norm:  {pred_m11:.6f}")
    print(f"  [0, 255] norm: {pred_255:.6f}")

    # Create a "Smooth" image
    img_smooth = np.ones((256, 256, 3), dtype=np.uint8) * 128
    p_smooth = d.preprocess_face(img_smooth)
    pred_smooth = d.model.predict(p_smooth, verbose=0)[0][0]
    print(f"Prediction for Smooth image ([0, 1]): {pred_smooth:.6f}")

if __name__ == "__main__":
    test_preprocessing()
