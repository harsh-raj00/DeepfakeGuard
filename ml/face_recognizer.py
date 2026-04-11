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
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("[INFO] face_recognition library available (dlib)")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

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
        self.tolerance = tolerance or config.FACE_RECOGNITION_TOLERANCE
        self.known_encodings = {}  # {username: [list of encoding vectors]}
        self.encodings_dir = config.ENCODINGS_DIR
        self._lock = Lock()

        # ── MediaPipe FaceMesh for fallback recognition ──
        self._face_mesh = None
        if not FACE_RECOGNITION_AVAILABLE and MEDIAPIPE_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            # Use higher tolerance for MediaPipe encoding (different scale)
            self.tolerance = 0.45
            print("[INFO] Face Recognition: Using MediaPipe + HOG encoding.")

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
        1. MediaPipe FaceMesh landmark geometry (inter-landmark distances)
        2. HOG (Histogram of Oriented Gradients) from the face region

        Returns a combined 196-d vector that is consistent for the same face.
        """
        if self._face_mesh is None:
            return None

        # Resize to standard size for consistency
        face_resized = cv2.resize(face_image, (128, 128))
        rgb_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # ── 1. Get MediaPipe landmarks ──
        with self._lock:
            results = self._face_mesh.process(rgb_face)

        if not results.multi_face_landmarks:
            # If no face mesh detected, use HOG only
            return self._hog_encoding(face_resized)

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

    def _hog_encoding(self, face_image, size=196):
        """
        Generate HOG-based encoding from face image.
        This is deterministic — same face, same lighting → same descriptor.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        # Apply histogram equalization for lighting invariance
        gray = cv2.equalizeHist(gray)

        # Compute HOG descriptor using OpenCV
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        descriptor = hog.compute(gray)

        if descriptor is None:
            return np.zeros(size, dtype=np.float64)

        descriptor = descriptor.flatten()

        # Reduce dimensionality by averaging blocks
        if len(descriptor) > size:
            # Chunk and average
            chunk_size = len(descriptor) // size
            reduced = np.array([
                np.mean(descriptor[i * chunk_size:(i + 1) * chunk_size])
                for i in range(size)
            ])
            descriptor = reduced

        # Pad if needed
        if len(descriptor) < size:
            descriptor = np.pad(descriptor, (0, size - len(descriptor)))

        # L2 normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        return descriptor[:size].astype(np.float64)

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
        best_distance = float('inf')

        for username, stored_encodings in self.known_encodings.items():
            for stored_enc in stored_encodings:
                # Ensure same dimension before comparing
                if len(encoding) != len(stored_enc):
                    continue
                distance = np.linalg.norm(encoding - stored_enc)
                if distance < best_distance:
                    best_distance = distance
                    best_match = username

        if best_distance <= self.tolerance and best_match is not None:
            return {
                "matched": True,
                "username": best_match,
                "distance": float(best_distance),
                "confidence": float(max(0, 1.0 - best_distance))
            }
        else:
            return {
                "matched": False,
                "username": "Unknown",
                "distance": float(best_distance),
                "confidence": float(max(0, 1.0 - best_distance))
            }

    def verify_user(self, username, face_image):
        """
        Verify that a face belongs to a specific known user (1:1 matching).

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

        # Compare against this user's stored encodings only
        distances = []
        for stored in self.known_encodings[username]:
            if len(encoding) == len(stored):
                distances.append(np.linalg.norm(encoding - stored))

        if not distances:
            return {
                "matched": False,
                "distance": 1.0,
                "confidence": 0.0,
                "message": "Encoding dimension mismatch."
            }

        min_distance = min(distances)

        return {
            "matched": min_distance <= self.tolerance,
            "distance": float(min_distance),
            "confidence": float(max(0, 1.0 - min_distance)),
            "message": "Match!" if min_distance <= self.tolerance else "No match."
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
