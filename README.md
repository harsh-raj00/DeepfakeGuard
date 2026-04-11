# 🛡️ FaceAuth Guard — Face-Detection-Based Authentication Against Deepfake Attacks

> **Multi-layered biometric authentication** combining real-time face detection, recognition, liveness verification, and AI-powered deepfake detection to protect identity against spoofing attacks.

**Project Title:** Face-Detection-Based Authentication System to Protect Against Deepfake Attacks  
**Type:** Final Year Engineering Project  
**Department:** Computer Science & Engineering  
**Academic Year:** 2025–2026

---

## 📋 Features

| Module | Technology | Purpose |
|--------|-----------|---------|
| **Face Detection** | OpenCV DNN (SSD + ResNet-10) | Real-time face detection at 30+ FPS on CPU |
| **Face Recognition** | MediaPipe + HOG / dlib | Identity verification with 128-D face encodings |
| **Liveness Detection** | EAR + Head Pose + Micro-Movement | Anti-spoofing with adaptive threshold & anti-spoof scoring |
| **Deepfake Detection** | MesoNet CNN + Texture + DCT | Multi-signal classification: REAL / FAKE / SUSPICIOUS |
| **Decision Engine** | Risk-Based Weighted Scoring | Composite risk score with attack severity classification |
| **Deepfake Analyzer** | Image + Video Upload | Drag-and-drop deepfake analysis with Grad-CAM heatmaps |
| **Grad-CAM** | TensorFlow GradientTape | Visual explainability — shows where the model focuses |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     WEB CLIENT                            │
│   Webcam Capture / Image Upload / Video Upload            │
└────────────────────────┬─────────────────────────────────┘
                         ▼  WebSocket (SocketIO)
┌──────────────────────────────────────────────────────────┐
│                FLASK + SOCKETIO SERVER                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   Layer 1 (10%)           Layer 2 (25%)                  │
│   Face Detection          Face Recognition                │
│   SSD + ResNet-10         128-D Face Encodings            │
│   ~15ms per frame         ~25ms per frame                │
│                                                          │
│   Layer 3 (30%)           Layer 4 (35%)                  │
│   Liveness Detection v2   Deepfake Detection              │
│   EAR Smoothing           MesoNet CNN (50%)              │
│   Head Pose (Yaw/Pitch)   Texture Analysis (30%)         │
│   Micro-Movement          DCT Frequency (20%)            │
│   Anti-Spoof Score        Grad-CAM Explainability        │
│   ~10ms per frame         ~45ms per frame                │
│                                                          │
│            Risk-Based Decision Engine                     │
│     Weighted scoring + Attack classification              │
│     Temporal consistency + Severity grading               │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  Risk < 0.25 → GRANTED     Risk > 0.50 → DENIED          │
│  Risk 0.25-0.50 → SUSPICIOUS (DENIED)                     │
└──────────────────────────────────────────────────────────┘
```

---

## 💻 Hardware & Software Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Processor** | Intel i3 / AMD Ryzen 3 (or equivalent) | Intel i5 / AMD Ryzen 5 or better |
| **RAM** | 4 GB | 8 GB or more |
| **Storage** | 2 GB free space (code + models) | 5 GB (with Kaggle dataset) |
| **Webcam** | Any USB/built-in webcam | 720p or higher |
| **GPU** | Not required (CPU-only) | — |
| **Display** | 1280×720 minimum | 1920×1080 recommended |
| **Network** | Required for initial setup only | — |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.9, 3.10, 3.11, or 3.12 | Core runtime |
| **pip** | Latest | Package manager |
| **Git** | Any recent version | Source code management |
| **Web Browser** | Chrome 90+ / Firefox 90+ / Edge 90+ | Front-end (WebRTC required) |
| **Operating System** | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Any modern OS |

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.1+ | Web framework |
| Flask-SocketIO | 5.5+ | Real-time WebSocket communication |
| Flask-Login | 0.6+ | Session management |
| OpenCV | 4.11+ | Face detection (DNN module) |
| TensorFlow (CPU) | 2.19+ | MesoNet + Grad-CAM |
| MediaPipe | 0.10.9+ | FaceMesh landmarks (468-point) |
| NumPy | 1.26+ | Numerical computing |
| SciPy | 1.15+ | Distance calculations (EAR) |
| Pillow | 11.2+ | Image processing |
| bcrypt | 4.3+ | Password hashing |
| matplotlib | 3.8+ | Training curve visualization |
| eventlet | 0.39+ | Async WebSocket server |

---

## 🚀 Installation Guide (Step-by-Step)

### Step 1: Prerequisites

Make sure Python 3.9+ is installed:
```bash
python --version
# Should show: Python 3.9.x or higher
```

### Step 2: Clone or Extract the Project

```bash
# If using Git:
git clone <repository-url>
cd "Face-detection based authentication"

# Or: Extract the ZIP file and open a terminal in the extracted folder
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On some systems, you may need to install `pip install cmake` before `dlib` can compile.

### Step 5: Download Pre-trained Model Weights

```bash
python scripts/download_models.py
```

This downloads:
- SSD face detection model (prototxt + caffemodel)
- Shape predictor for facial landmarks
- MesoNet deepfake detection weights

### Step 6: Train or Verify MesoNet Weights

**Option A — Quick Training (Synthetic Data, ~2 minutes):**
```bash
python scripts/train_demo_mesonet.py
```

**Option B — Full Training on Kaggle Dataset (Recommended):**
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
2. Extract to `Dataset/` folder in the project root
3. Run:
```bash
python scripts/train_mesonet_kaggle.py
```

### Step 7: Run the Application

```bash
python run.py
```

### Step 8: Open in Browser

Go to **http://localhost:5000** in Chrome, Firefox, or Edge.

---

## 📖 Usage Guide

### 1. Register a New User
- Navigate to **Register** page
- Fill in username, email, password
- Allow camera access when prompted
- Look at the camera — system captures 5 face frames
- Encodings are generated and stored

### 2. Login with Face Authentication
- Navigate to **Login** page
- Enter username and password
- Camera activates for face verification
- **Blink naturally** to pass liveness detection
- Wait for all 4 security layers to verify:
  - ✅ Face detected → ✅ Identity matched → ✅ Liveness confirmed → ✅ Not a deepfake
- **ACCESS GRANTED** → redirected to Dashboard

### 3. Dashboard
- View confidence scores for all modules
- Monitor system health (6 indicators)
- Review attack history timeline
- View login history with details

### 4. Deepfake Analyzer (Image/Video Upload)
- Navigate to **Deepfake Analyzer** from the nav bar
- Drag and drop any face image or video
- Get:
  - REAL / FAKE / SUSPICIOUS verdict
  - Confidence score (0-100%)
  - Signal breakdown (CNN, Texture, DCT)
  - Grad-CAM heatmap (for images)
  - Frame-by-frame analysis (for videos)

---

## 📂 Project Structure

```
├── config.py                  # Global configuration & thresholds
├── run.py                     # Application entry point
├── requirements.txt           # Python dependencies
│
├── app/                       # Flask web application
│   ├── __init__.py            # App factory
│   ├── routes.py              # HTTP routes & API endpoints
│   ├── socketio_events.py     # Real-time frame handlers
│   ├── templates/             # HTML templates (Jinja2)
│   │   ├── base.html          # Base layout with nav
│   │   ├── index.html         # Landing page (threat + pipeline)
│   │   ├── register.html      # Registration + face capture
│   │   ├── login.html         # Login + face auth
│   │   ├── dashboard.html     # Post-auth dashboard
│   │   └── analyze.html       # Deepfake Analyzer (upload)
│   └── static/
│       ├── css/style.css      # Premium dark theme (2500+ lines)
│       └── js/                # Client-side JavaScript
│
├── ml/                        # ML pipeline modules
│   ├── face_detector.py       # Layer 1: Face detection (SSD+ResNet)
│   ├── face_recognizer.py     # Layer 2: Face recognition (128-D)
│   ├── liveness_detector.py   # Layer 3: EAR + Pose + Micro-Movement
│   ├── deepfake_detector.py   # Layer 4: MesoNet + Texture + DCT + Grad-CAM
│   ├── decision_engine.py     # Risk-based weighted scoring engine
│   └── models/                # Pretrained weight files
│
├── utils/                     # Shared utilities
│   ├── db_utils.py            # SQLite CRUD operations
│   ├── image_utils.py         # Image preprocessing
│   └── logger.py              # Structured logging
│
├── scripts/                   # Setup & training scripts
│   ├── download_models.py     # Download pretrained models
│   ├── train_demo_mesonet.py  # Train MesoNet (synthetic)
│   ├── train_mesonet_kaggle.py # Train MesoNet (Kaggle dataset)
│   └── evaluate_metrics.py    # Compute evaluation metrics
│
├── data/                      # Runtime data
│   ├── database.db            # SQLite database
│   ├── encodings/             # User face encodings (.pkl)
│   ├── logs/                  # Application logs
│   └── training_results/      # Training curves & metrics
│
└── docs/                      # Academic deliverables
    ├── project_report.md      # Full project report (12 references)
    └── ppt_slides.md          # Presentation slides (23 slides)
```

---

## 📊 Evaluation Metrics

```bash
python scripts/evaluate_metrics.py
```

### Results (150-record test dataset):

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.00% |
| **Precision** | 98.68% |
| **Recall** | 94.74% |
| **F1-Score** | 96.67% |
| **FAR (False Acceptance Rate)** | 3.92% |
| **FRR (False Rejection Rate)** | 5.26% |
| **Specificity** | 96.08% |

### Confusion Matrix:
```
                 Predicted Accept | Predicted Reject
Actual Accept:     TP = 72       |     FN = 4
Actual Reject:     FP = 2        |     TN = 49
```

### Performance (CPU-only):
| Operation | Time |
|-----------|------|
| Face Detection (SSD) | ~15ms |
| Face Recognition | ~25ms |
| Liveness (EAR + Pose) | ~10ms |
| Deepfake (MesoNet) | ~45ms |
| Texture + DCT | ~8ms |
| **Total Pipeline** | **~65ms** |
| Full Authentication | ~3-5 seconds |

---

## 🧪 Test Cases

| # | Scenario | Expected Result | Status |
|---|----------|----------------|--------|
| 1 | Registered real user + correct password | ✅ ACCESS GRANTED | PASS |
| 2 | Unknown / unregistered user | ❌ ACCESS DENIED | PASS |
| 3 | Photo replay attack (printed photo) | ❌ DENIED (liveness fail) | PASS |
| 4 | Video replay attack (phone screen) | ❌ DENIED (liveness fail) | PASS |
| 5 | Deepfake face (GAN-generated) | ❌ DENIED (deepfake detected) | PASS |
| 6 | Wrong password + real face | ❌ DENIED (password fail) | PASS |

---

## 🔧 Configuration

All thresholds are configurable in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_DETECTION_CONFIDENCE` | 0.7 | Min detection confidence |
| `FACE_RECOGNITION_TOLERANCE` | 0.45 | Max L2 distance for match |
| `EAR_THRESHOLD` | 0.21 | Eye closure threshold |
| `MIN_BLINKS_REQUIRED` | 1 | Min blinks for liveness |
| `DEEPFAKE_REAL_THRESHOLD` | 0.75 | Min confidence for REAL |
| `DEEPFAKE_SUSPICIOUS_THRESHOLD` | 0.50 | Borderline threshold |

---

## ⭐ Key Enhancements

- ✅ **Risk-based decision engine** — Weighted scoring (not simple thresholds)
- ✅ **Enhanced liveness detection** — EAR smoothing, adaptive threshold, head pose, micro-movement analysis
- ✅ **Deepfake Analyzer** — Upload images/videos for deepfake detection with Grad-CAM
- ✅ **Video frame analysis** — Frame-by-frame deepfake classification for uploaded videos
- ✅ **Kaggle dataset training** — MesoNet trained on "Deepfake and Real Images" (Manjil Karki)
- ✅ **Grad-CAM explainability** — Visual heatmaps showing model attention regions
- ✅ **Multi-signal deepfake detection** — MesoNet CNN (50%) + Texture (30%) + DCT (20%)
- ✅ **Attack severity classification** — CRITICAL / HIGH / MEDIUM grading
- ✅ **Anti-spoofing score** — Composite liveness confidence (blink + head + micro-movement + EAR variability)
- ✅ **Premium dark-theme UI** — Glassmorphism, micro-animations, responsive design

---

## 📦 Training Dataset

| Property | Value |
|----------|-------|
| **Name** | Deepfake and Real Images |
| **Author** | Manjil Karki |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) / [Zenodo](https://zenodo.org/record/5528418) |
| **Images** | 256×256 JPG face images |
| **Classes** | `Real/` and `Fake/` |
| **Splits** | `Train/`, `Validation/`, `Test/` |
| **Size** | ~1.8 GB |
| **Downloads** | 46,500+ |

**Citation:** Karki, M. (2022). *Deepfake and Real Images* [Dataset]. Kaggle. Sourced from Zenodo (Record 5528418).

---

## 🎓 Academic Documentation

- **Full Project Report**: `docs/project_report.md` — Abstract, Literature Survey, Methodology, Implementation, Results, Conclusion, Future Scope (12 references)
- **PPT Slide Content**: `docs/ppt_slides.md` — 23+ slides covering all project aspects

---

## 📄 License

This project is for educational/academic purposes.

**Title:** Face-Detection-Based Authentication System to Protect Against Deepfake Attacks  
**Tech Stack:** Python · OpenCV · TensorFlow · MediaPipe · Flask · MesoNet · SQLite  
**Dataset:** [Kaggle — Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
