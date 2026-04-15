"""Debug script to expose exactly what the DeepfakeDetector outputs on different inputs."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from ml.deepfake_detector import DeepfakeDetector

d = DeepfakeDetector()

# ── Test 1: Smooth/plasticised face (typical GAN/deepfake look) ──
fake_sim = np.ones((256, 256, 3), dtype=np.uint8) * 150
fake_sim = cv2.GaussianBlur(fake_sim, (21, 21), 8)
fake_sim += np.random.randint(0, 3, (256, 256, 3), dtype=np.uint8)

# ── Test 2: Natural face-like (natural texture, noise, edges) ──
real_sim = np.zeros((256, 256, 3), dtype=np.uint8)
for y in range(256):
    for x in range(256):
        real_sim[y, x] = [
            int(120 + 30*np.sin(x/20.0)*np.cos(y/20.0)),
            int(100 + 25*np.cos(x/15.0)),
            int(90 + 20*np.sin(y/18.0))
        ]
real_sim = np.clip(real_sim, 0, 255).astype(np.uint8)
real_sim += np.random.randint(0, 20, (256, 256, 3), dtype=np.uint8)
real_sim = np.clip(real_sim, 0, 255).astype(np.uint8)

print("=" * 60)
print("REAL SIM SCORES:")
r_tex = d._texture_score(real_sim)
r_freq = d._frequency_score(real_sim)
r_meso = d._mesonet_score(real_sim)
print(f"  texture:   {r_tex:.4f}")
print(f"  frequency: {r_freq:.4f}")
print(f"  mesonet:   {r_meso}")
r = d.analyze_face(real_sim)
print(f"  FINAL: conf_real={r['confidence_real']:.4f}  label={r['label']}")

print()
print("FAKE SIM SCORES:")
f_tex = d._texture_score(fake_sim)
f_freq = d._frequency_score(fake_sim)
f_meso = d._mesonet_score(fake_sim)
print(f"  texture:   {f_tex:.4f}")
print(f"  frequency: {f_freq:.4f}")
print(f"  mesonet:   {f_meso}")
f = d.analyze_face(fake_sim)
print(f"  FINAL: conf_real={f['confidence_real']:.4f}  label={f['label']}")

print()
print(f"Thresholds: REAL>={d.real_threshold}  SUSPICIOUS>={d.suspicious_threshold}")
print("=" * 60)
