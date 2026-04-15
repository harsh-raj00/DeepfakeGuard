import sys
import os

print("Testing MediaPipe initialization...")
try:
    import mediapipe as mp
    print("Mediapipe imported.")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    print("FaceMesh initialized.")
except Exception as e:
    print("Error:", e)

print("Testing MesoNet TF model...")
try:
    from ml.deepfake_detector import DeepfakeDetector
    det = DeepfakeDetector()
    print("DeepfakeDetector initialized.")
except Exception as e:
    print("Error:", e)
