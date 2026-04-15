"""
=============================================================================
 Module 2: Face Recognition — MediaPipe + Histogram-of-Oriented-Gradients
 Generates face embeddings using a combination approach:
   1. MediaPipe FaceMesh landmarks → geometry-based descriptor
   2. HOG features from the face region → texture-based descriptor
 The combined descriptor is used for identity matching.
 Falls back to face_recognition/dlib if available.
=============================================================================
"""

import os
import sys
import pickle
import cv2
import numpy as np
from threading import Lock

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Try face_recognition (dlib-based) first, fall back to MediaPipe ─────────
import sys

FACE_RECOGNITION_AVAILABLE = False
FACE_RECOGNITION_MODELS_AVAILABLE = False

# We must prevent face_recognition from permanently closing sys.stdin if its internal imports fail
_original_close = getattr(sys.stdin, 'close', None)
if _original_close is not None:
    sys.stdin.close = lambda: None

try:
    import face_recognition
    import face_recognition_models  # type: ignore
    FACE_RECOGNITION_AVAILABLE = True
    FACE_RECOGNITION_MODELS_AVAILABLE = True
    print("[INFO] face_recognition library available (dlib)")
except (Exception, SystemExit) as e:
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_MODELS_AVAILABLE = False
    print(f"[WARNING] face_recognition failed to initialize: {e}")
finally:
    if _original_close is not None:
        sys.stdin.close = _original_close

# ── Try MediaPipe as fallback ───────────────────────────────────────────────
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("[INFO] MediaPipe available for face recognition fallback.")
except ImportError:
    pass

if not FACE_RECOGNITION_AVAILABLE and not MEDIAPIPE_AVAILABLE:
    print("[WARNING] Neither face_recognition nor MediaPipe available. "
          "Face recognition will use basic mode.")


class FaceRecognizer:
    """
    Face recognition engine.
    Primary: face_recognition library (128-d dlib encodings)
    Fallback: MediaPipe landmarks + HOG descriptor (deterministic)
    """

    def __init__(self, tolerance=None):
        """
        Initialize face recognizer and load existing encodings.

        Args:
            tolerance (float): Maximum distance for a positive match.
                              Lower = stricter. Defaults to config value.
        """
        self.known_encodings = {}  # {username: [list of encoding vectors]}
        self.encodings_dir = config.ENCODINGS_DIR
        self._lock = Lock()

        # ── Matching mode: cosine similarity vs L2 distance ──
        # dlib produces 128-d encodings where L2 < 0.6 = match.
        # HOG produces 8100-d L2-normalized vectors where L2 distances are
        # always 0.5–0.8 even for the SAME face, making L2 useless.
        # For HOG, we use cosine similarity (0–1, higher = better match).
        self.use_cosine = not FACE_RECOGNITION_AVAILABLE
        if FACE_RECOGNITION_AVAILABLE:
            self.tolerance = tolerance or config.FACE_RECOGNITION_TOLERANCE
            self.cosine_threshold = 0.0  # unused
        else:
            self.tolerance = 1.0  # unused for cosine mode
            self.cosine_threshold = 0.75  # cosine similarity >= 0.75 = match

        # ── MediaPipe FaceMesh for fallback recognition ──
        self._face_mesh = None
        if not FACE_RECOGNITION_AVAILABLE and MEDIAPIPE_AVAILABLE:
            try:
                # Use the standard mp.solutions path (same as liveness_detector)
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
                print("[INFO] Face Recognition: Using MediaPipe + HOG encoding.")
                self.cosine_threshold = 0.80  # Geometry+HOG is more discriminative
            except Exception as e:
                print(f"[WARNING] MediaPipe face_mesh unavailable for recognition: {e}")
                print("[WARNING] Falling back to HOG-only mode (reduced accuracy).")
                self.cosine_threshold = 0.75  # HOG-only needs more leniency
        elif not FACE_RECOGNITION_AVAILABLE:
            self.cosine_threshold = 0.75
            print("[WARNING] HOG-only mode active. Using cosine similarity >= 0.75.")

        print(f"[INFO] Face Recognition mode: {'dlib' if FACE_RECOGNITION_AVAILABLE else 'HOG-cosine'}, "
              f"threshold: {self.tolerance if FACE_RECOGNITION_AVAILABLE else self.cosine_threshold}")

        # Load any previously stored encodings
        self._load_all_encodings()

    def _load_all_encodings(self):
        """Load all stored user encodings from disk."""
        if not os.path.exists(self.encodings_dir):
            os.makedirs(self.encodings_dir, exist_ok=True)
            return

        for fname in os.listdir(self.encodings_dir):
            if fname.endswith(".pkl"):
                username = fname.replace(".pkl", "")
                filepath = os.path.join(self.encodings_dir, fname)
                try:
                    with open(filepath, "rb") as f:
                        self.known_encodings[username] = pickle.load(f)
                    print(f"[INFO] Loaded encodings for user: {username}")
                except Exception as e:
                    print(f"[WARNING] Failed to load encodings for {username}: {e}")

    def generate_encoding(self, face_image):
        """
        Generate a face encoding from a cropped face image.

        Uses face_recognition/dlib if available, otherwise falls back
        to MediaPipe landmarks + HOG features.

        Args:
            face_image (np.ndarray): Cropped face image (BGR format).

        Returns:
            np.ndarray or None: Encoding vector, or None if failed.
        """
        if face_image is None or face_image.size == 0:
            return None

        # ── Primary: dlib/face_recognition ──
        if FACE_RECOGNITION_AVAILABLE:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(
                rgb_image,
                num_jitters=config.FACE_RECOGNITION_NUM_JITTERS
            )
            if len(encodings) > 0:
                return encodings[0]
            return None

        # ── Fallback: MediaPipe landmarks + HOG descriptor ──
        return self._generate_mediapipe_encoding(face_image)

    def _generate_mediapipe_encoding(self, face_image):
        """
        Generate a deterministic face encoding using:
        1. MediaPipe FaceMesh landmark geometry (inter-landmark distances) if available
        2. HOG (Histogram of Oriented Gradients) from the face region

        Returns a combined 196-d vector that is consistent for the same face.
        """
        # Resize to standard size for consistency
        face_resized = cv2.resize(face_image, (128, 128))
        rgb_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        if self._face_mesh is None:
            # FALLBACK FOR PYTHON 3.13: Use HOG-only encoding if MediaPipe fails.
            hog_desc = self._hog_encoding(face_resized, size=196)
            return hog_desc

        # ── 1. Get MediaPipe landmarks ──
        with self._lock:
            results = self._face_mesh.process(rgb_face)

        if not results or not results.multi_face_landmarks:
            # If no face mesh detected, use HOG only
            return self._hog_encoding(face_resized, size=196)

        landmarks = results.multi_face_landmarks[0]
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])

        # ── 2. Geometry descriptor: normalized inter-landmark distances ──
        # Use key points: eyes, nose, mouth, jawline (68 key indices)
        key_indices = [
            1, 4, 5, 6, 10, 33, 46, 52, 53, 54, 55, 56, 57, 58, 61,
            67, 93, 103, 104, 105, 132, 133, 136, 145, 153, 155, 157,
            158, 159, 160, 161, 162, 163, 168, 173, 188, 196, 197, 234,
            263, 276, 282, 283, 284, 285, 286, 287, 288, 291, 297, 323,
            332, 333, 334, 361, 362, 365, 373, 380, 382, 384, 385, 386,
            387, 388, 389, 390, 397, 454
        ]

        # Compute pairwise distances between subset of key points
        key_points = coords[key_indices[:32]]  # Use 32 key points
        dists = []
        for i in range(len(key_points)):
            for j in range(i + 1, min(i + 4, len(key_points))):
                d = np.linalg.norm(key_points[i] - key_points[j])
                dists.append(d)

        # Normalize by face size (distance between outer eye corners)
        face_width = np.linalg.norm(coords[33] - coords[263])
        if face_width > 0:
            dists = [d / face_width for d in dists]

        geometry_desc = np.array(dists[:96], dtype=np.float64)
        # Pad if needed
        if len(geometry_desc) < 96:
            geometry_desc = np.pad(geometry_desc, (0, 96 - len(geometry_desc)))

        # ── 3. HOG descriptor from face ──
        hog_desc = self._hog_encoding(face_resized, size=100)

        # ── 4. Combine into single vector ──
        combined = np.concatenate([geometry_desc, hog_desc])

        # L2 normalize the combined descriptor
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors. Returns 0-1."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _hog_encoding(self, face_image, size=196):
        """
        Generate HOG-based encoding from face image.
        Includes face-centering and padding to stabilize against
        bounding box jitter from the Haar cascade detector.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # ── Stabilize the crop: add 10% padding and center the face ──
        # This reduces sensitivity to Haar cascade bounding box jitter
        h, w = gray.shape[:2]
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)
        # Crop inward to remove background edges that vary between frames
        crop_gray = gray[pad_y:h - pad_y, pad_x:w - pad_x]
        if crop_gray.shape[0] < 20 or crop_gray.shape[1] < 20:
            crop_gray = gray  # fallback if face is too small

        gray = cv2.resize(crop_gray, (128, 128))

        # Apply CLAHE for better lighting invariance than basic equalizeHist
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # ── Apply Gaussian blur to reduce noise sensitivity ──
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Compute HOG descriptor using OpenCV
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        descriptor = hog.compute(gray)

        if descriptor is None:
            return np.zeros(8100, dtype=np.float64)

        descriptor = descriptor.flatten()

        # L2 normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        return descriptor.astype(np.float64)

    def register_user(self, username, face_images):
        """
        Register a new user by capturing multiple face encodings.

        Args:
            username (str): Unique username/identifier.
            face_images (list): List of cropped face images (BGR).

        Returns:
            dict: Registration result with status and encoding count.
        """
        encodings = []
        for img in face_images:
            encoding = self.generate_encoding(img)
            if encoding is not None:
                encodings.append(encoding)

        if len(encodings) == 0:
            return {
                "success": False,
                "message": "Could not generate any face encodings.",
                "num_encodings": 0
            }

        # Store encodings in memory
        self.known_encodings[username] = encodings

        # Persist to disk
        filepath = os.path.join(self.encodings_dir, f"{username}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(encodings, f)

        mode = "dlib" if FACE_RECOGNITION_AVAILABLE else "MediaPipe+HOG"
        print(f"[INFO] Registered user '{username}' with {len(encodings)} encodings ({mode}).")
        return {
            "success": True,
            "message": f"Registered with {len(encodings)} face encodings.",
            "num_encodings": len(encodings)
        }

    def recognize_face(self, face_image):
        """
        Identify a face by comparing against all stored encodings.
        Uses cosine similarity for HOG mode, L2 distance for dlib mode.

        Args:
            face_image (np.ndarray): Cropped face image (BGR).

        Returns:
            dict: Recognition result.
        """
        encoding = self.generate_encoding(face_image)
        if encoding is None:
            return {
                "matched": False,
                "username": "Unknown",
                "distance": 1.0,
                "confidence": 0.0
            }

        best_match = None
        best_score = -1.0 if self.use_cosine else float('inf')

        for username, stored_encodings in self.known_encodings.items():
            scores = []
            for stored_enc in stored_encodings:
                if len(encoding) != len(stored_enc):
                    continue
                if self.use_cosine:
                    scores.append(self._cosine_similarity(encoding, stored_enc))
                else:
                    scores.append(np.linalg.norm(encoding - stored_enc))

            if not scores:
                continue

            scores.sort(reverse=self.use_cosine)  # Best first
            top_k = min(3, len(scores))
            avg_score = float(np.mean(scores[:top_k]))

            if self.use_cosine:
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = username
            else:
                if avg_score < best_score:
                    best_score = avg_score
                    best_match = username

        if self.use_cosine:
            matched = best_score >= self.cosine_threshold and best_match is not None
            return {
                "matched": matched,
                "username": best_match or "Unknown",
                "distance": float(1.0 - best_score),  # Convert to distance for API compat
                "confidence": float(max(0, best_score))
            }
        else:
            matched = best_score <= self.tolerance and best_match is not None
            return {
                "matched": matched,
                "username": best_match or "Unknown",
                "distance": float(best_score),
                "confidence": float(max(0, 1.0 - best_score))
            }

    def verify_user(self, username, face_image):
        """
        Verify that a face belongs to a specific known user (1:1 matching).
        Uses cosine similarity for HOG mode, L2 distance for dlib mode.
        Includes anti-confusion margin check against other stored users.

        Args:
            username (str): The claimed identity to verify against.
            face_image (np.ndarray): Cropped face image (BGR).

        Returns:
            dict: Verification result with 'matched', 'distance', 'confidence'.
        """
        if username not in self.known_encodings:
            return {
                "matched": False,
                "distance": 1.0,
                "confidence": 0.0,
                "message": f"User '{username}' not registered."
            }

        encoding = self.generate_encoding(face_image)
        if encoding is None:
            return {
                "matched": False,
                "distance": 1.0,
                "confidence": 0.0,
                "message": "Could not generate face encoding."
            }

        # Compare against this user's stored encodings
        scores = []
        for stored in self.known_encodings[username]:
            if len(encoding) == len(stored):
                if self.use_cosine:
                    scores.append(self._cosine_similarity(encoding, stored))
                else:
                    scores.append(np.linalg.norm(encoding - stored))

        if not scores:
            return {
                "matched": False,
                "distance": 1.0,
                "confidence": 0.0,
                "message": "Encoding dimension mismatch — please re-register your face."
            }

        # Average of best-K scores for robust verification
        scores.sort(reverse=self.use_cosine)  # Best first
        top_k = min(3, len(scores))
        avg_score = float(np.mean(scores[:top_k]))

        # ── Majority vote ──
        if self.use_cosine:
            within_threshold = sum(1 for s in scores[:top_k] if s >= self.cosine_threshold)
        else:
            within_threshold = sum(1 for s in scores[:top_k] if s <= self.tolerance)
        majority_needed = max(1, (top_k + 1) // 2)
        majority_ok = within_threshold >= majority_needed

        # ── Anti-confusion margin check ──
        best_other_score = -1.0 if self.use_cosine else float('inf')
        for other_user, other_encodings in self.known_encodings.items():
            if other_user == username:
                continue
            for stored in other_encodings:
                if len(encoding) == len(stored):
                    if self.use_cosine:
                        s = self._cosine_similarity(encoding, stored)
                        best_other_score = max(best_other_score, s)
                    else:
                        d = np.linalg.norm(encoding - stored)
                        best_other_score = min(best_other_score, d)

        margin_ok = True
        if self.use_cosine:
            # For cosine: claimed user should score higher than any other user
            if best_other_score > -1.0 and best_other_score >= avg_score:
                margin_ok = False
                print(f"[VERIFY] MARGIN FAIL: {username} cosine={avg_score:.4f}, "
                      f"best_other_cosine={best_other_score:.4f}")
        else:
            # For L2: claimed user should have smaller distance
            if best_other_score < float('inf'):
                margin = best_other_score - avg_score
                if margin < avg_score * 0.15:
                    margin_ok = False
                    print(f"[VERIFY] MARGIN FAIL: {username} avg_dist={avg_score:.4f}, "
                          f"best_other={best_other_score:.4f}")

        if self.use_cosine:
            matched = avg_score >= self.cosine_threshold and majority_ok and margin_ok
            distance_for_log = 1.0 - avg_score  # Convert for API compat
            confidence = avg_score
            print(f"[VERIFY] User={username}, cosine_sim={avg_score:.4f}, "
                  f"threshold={self.cosine_threshold:.4f}, majority={within_threshold}/{top_k}, "
                  f"margin_ok={margin_ok}, matched={matched}, "
                  f"all_scores={[f'{s:.4f}' for s in scores]}")
        else:
            matched = avg_score <= self.tolerance and majority_ok and margin_ok
            distance_for_log = avg_score
            confidence = max(0, 1.0 - avg_score)
            print(f"[VERIFY] User={username}, avg_dist={avg_score:.4f}, "
                  f"tolerance={self.tolerance:.4f}, majority={within_threshold}/{top_k}, "
                  f"margin_ok={margin_ok}, matched={matched}, "
                  f"all_dists={[f'{s:.4f}' for s in scores]}")

        return {
            "matched": matched,
            "username": username,
            "distance": float(distance_for_log),
            "confidence": float(max(0, confidence)),
            "message": "Match!" if matched else "No match."
        }

    def get_registered_users(self):
        """Return list of all registered usernames."""
        return list(self.known_encodings.keys())

    def delete_user(self, username):
        """Delete a user's encodings from memory and disk."""
        if username in self.known_encodings:
            del self.known_encodings[username]
            filepath = os.path.join(self.encodings_dir, f"{username}.pkl")
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"[INFO] Deleted user '{username}'.")
            return True
        return False


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    print(f"[INFO] Registered users: {recognizer.get_registered_users()}")
    print(f"[INFO] Using dlib: {FACE_RECOGNITION_AVAILABLE}")
    print(f"[INFO] Using MediaPipe: {MEDIAPIPE_AVAILABLE}")
