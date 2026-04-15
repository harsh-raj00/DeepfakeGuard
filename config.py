"""
=============================================================================
 Global Configuration — Face-Detection-Based Authentication System
 All thresholds, paths, and system-wide settings live here.
=============================================================================
"""

import os

# ── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
ENCODINGS_DIR = os.path.join(DATA_DIR, "encodings")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DATABASE_PATH = os.path.join(DATA_DIR, "database.db")

# ── Create directories if they don't exist ──────────────────────────────────
for d in [DATA_DIR, MODELS_DIR, ENCODINGS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Flask Configuration ────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "deepfake-auth-secret-key-change-in-prod")
DEBUG = True

# ── Face Detection (OpenCV DNN — SSD/ResNet-10) ────────────────────────────
FACE_DETECTION_CONFIDENCE = 0.7        # Minimum confidence to accept a detected face
FACE_DETECTION_MODEL_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_DETECTION_MODEL_WEIGHTS = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ── Face Recognition (dlib / face_recognition) ─────────────────────────────
FACE_RECOGNITION_TOLERANCE = 0.45      # Max L2 distance for a match (lower = stricter)
FACE_RECOGNITION_NUM_JITTERS = 1       # Number of re-samples for encoding (1 = fast)
REGISTRATION_MIN_FRAMES = 5            # Min frames to capture during registration

# ── Liveness Detection (EAR + Head Movement) ───────────────────────────────
EAR_THRESHOLD = 0.19                   # Below this = eye closed (lowered for smoothing)
EAR_CONSEC_FRAMES = 1                  # Consecutive frames below threshold = 1 blink
MIN_BLINKS_REQUIRED = 1               # Minimum blinks needed in auth window
HEAD_MOVEMENT_THRESHOLD = 3.0         # Pixels of nose movement to count as head motion
SHAPE_PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

# ── Deepfake Detection (MesoNet) ───────────────────────────────────────────
DEEPFAKE_MODEL_PATH = os.path.join(MODELS_DIR, "mesonet.weights.h5")
DEEPFAKE_INPUT_SIZE = (256, 256)       # MesoNet expected input size
DEEPFAKE_REAL_THRESHOLD = 0.60        # Confidence >= this to classify as REAL
DEEPFAKE_SUSPICIOUS_THRESHOLD = 0.40  # Between this and REAL = suspicious

# ── Decision Engine ─────────────────────────────────────────────────────────
AUTH_WINDOW_FRAMES = 30                # Sliding window size (~1 sec at 30fps)
FACE_DETECT_EVERY_N = 1               # Run face detection on every frame
FACE_RECOG_EVERY_N = 5                # Run recognition every 5th frame
DEEPFAKE_EVERY_N = 10                 # Run deepfake check every 10th frame

# ── Webcam ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0                       # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ── Model Download URLs ────────────────────────────────────────────────────
MODEL_URLS = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "shape_predictor_68_face_landmarks.dat.bz2": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
}

# ── Kaggle Dataset Configuration ───────────────────────────────────────────
# Dataset: "Deepfake and Real Images" by Manjil Karki
# URL: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
# Structure: Dataset/Train/{Real,Fake}  Dataset/Validation/{Real,Fake}  Dataset/Test/{Real,Fake}
KAGGLE_DATASET_PATH = os.path.join(BASE_DIR, "Dataset")  # Path to extracted Kaggle dataset

# ── Training Hyperparameters ───────────────────────────────────────────────
TRAINING_BATCH_SIZE = 32
TRAINING_EPOCHS = 15
TRAINING_LEARNING_RATE = 1e-3
TRAINING_IMAGE_SIZE = (256, 256)

# ── Training Outputs ──────────────────────────────────────────────────────
TRAINING_RESULTS_DIR = os.path.join(DATA_DIR, "training_results")
GRADCAM_OUTPUT_DIR = os.path.join(DATA_DIR, "gradcam_outputs")
os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)
