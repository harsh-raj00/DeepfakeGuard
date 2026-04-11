"""
=============================================================================
 Module 3: Liveness Detection — Enhanced EAR + Head Movement + Micro-Texture
 Uses MediaPipe FaceMesh for 468-point face landmarks.
 
 Improvements over basic EAR:
  - Dual-path blink detection (raw EAR + smoothed EAR)
  - Rapid EAR drop detection (relative change from recent baseline)
  - EAR temporal smoothing (exponential moving average)
  - Multi-frame blink validation window (lenient for 8 FPS)
  - Adaptive EAR threshold based on user's open-eye baseline
  - Head pose estimation (yaw/pitch) via facial geometry
  - Micro-movement analysis (static frame detection)
  - Anti-spoofing score output for decision engine
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
from collections import deque
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from scipy.spatial import distance as dist

# ── MediaPipe FaceMesh landmark indices ──────────────────────────────────────
# Right eye: 6 landmarks (top-to-bottom ordering for EAR)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Left eye: 6 landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Additional eye indices for more robust EAR (outer ring)
RIGHT_EYE_UPPER = [159, 160, 161]
RIGHT_EYE_LOWER = [144, 145, 153]
LEFT_EYE_UPPER  = [386, 385, 384]
LEFT_EYE_LOWER  = [380, 374, 373]

NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
FOREHEAD_INDEX = 10
LEFT_CHEEK_INDEX = 234
RIGHT_CHEEK_INDEX = 454

# Additional landmarks for more movement tracking
LEFT_EAR_INDEX = 234      # Near left ear
RIGHT_EAR_INDEX = 454     # Near right ear
LEFT_TEMPLE_INDEX = 127
RIGHT_TEMPLE_INDEX = 356


class LivenessDetector:
    """
    Enhanced liveness detector using MediaPipe FaceMesh:
    1. Dual-path blink detection (raw + smoothed EAR with rapid-drop)
    2. Adaptive baseline EAR threshold
    3. Lenient blink validation for 8 FPS webcam streams
    4. Head pose estimation via facial geometry
    5. Micro-movement analysis (detects static presentations)
    6. Outputs an anti-spoofing confidence score (0-1)
    """

    def __init__(self):
        """Initialize the enhanced liveness detector."""
        self.ear_threshold = config.EAR_THRESHOLD
        self.ear_consec_frames = config.EAR_CONSEC_FRAMES
        self.min_blinks = config.MIN_BLINKS_REQUIRED
        self.head_movement_threshold = config.HEAD_MOVEMENT_THRESHOLD

        # ── State tracking ──
        self.blink_counter = 0
        self.total_blinks = 0
        self.ear_history = deque(maxlen=90)        # Longer history for smoothing
        self.raw_ear_history = deque(maxlen=30)     # Raw (unsmoothed) EAR history
        self.nose_positions = deque(maxlen=30)
        self.head_movement_detected = False
        self._lock = Lock()

        # ── Enhanced: EAR smoothing ──
        self.smoothed_ear = None
        self.ear_alpha = 0.5  # Higher alpha = more responsive to real blinks

        # ── Enhanced: Adaptive baseline ──
        self.baseline_ears = deque(maxlen=50)
        self.baseline_established = False
        self.adaptive_threshold = self.ear_threshold

        # ── Enhanced: Blink validation (lenient for 8 FPS) ──
        self.blink_candidate_start = None
        self.blink_candidate_duration = 0
        self.min_blink_duration = 1    # Allow single-frame blinks at 8 FPS
        self.max_blink_duration = 25   # Accommodate slow blinks

        # ── Enhanced: Raw EAR blink tracking (second path) ──
        self.raw_blink_counter = 0

        # ── Enhanced: Rapid-drop detection ──
        self.recent_open_ear = deque(maxlen=15)   # Recent open-eye EAR values
        self.drop_threshold_pct = 0.25             # 25% drop from recent avg = blink

        # ── Enhanced: Micro-movement analysis ──
        self.landmark_history = deque(maxlen=20)
        self.static_frame_count = 0
        self.static_threshold = 0.8   # Lower = more sensitive to static detection

        # ── Enhanced: Head pose ──
        self.head_yaw_history = deque(maxlen=20)
        self.head_pitch_history = deque(maxlen=20)

        # ── Anti-spoofing score components ──
        self.spoof_scores = {
            'blink_naturalness': 0.0,
            'head_dynamism': 0.0,
            'micro_movement': 0.0,
            'ear_variability': 0.0
        }

        # ── Initialize MediaPipe FaceMesh ──
        self.face_mesh = None
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,   # Lower for better tracking
                min_tracking_confidence=0.4     # Lower for better continuity
            )
            print("[INFO] Liveness Detection: MediaPipe FaceMesh loaded (enhanced v2).")
        except Exception as e:
            print(f"[WARNING] MediaPipe not available: {e}")
            print("[WARNING] Liveness detection will use fallback mode.")

    @staticmethod
    def calculate_ear(eye_points):
        """
        Calculate the Eye Aspect Ratio (EAR) for a single eye.

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Args:
            eye_points (list): 6 (x, y) points for one eye.

        Returns:
            float: Eye Aspect Ratio value.
        """
        eye = np.array(eye_points, dtype=np.float64)
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear

    @staticmethod
    def calculate_ear_extended(landmarks, upper_indices, lower_indices, corner_indices):
        """
        Calculate EAR using multiple upper/lower lid points for robustness.
        Averages vertical distances across 3 upper and 3 lower landmarks.
        """
        upper = np.array([landmarks[i] for i in upper_indices], dtype=np.float64)
        lower = np.array([landmarks[i] for i in lower_indices], dtype=np.float64)
        corners = np.array([landmarks[i] for i in corner_indices], dtype=np.float64)

        # Average vertical distances
        vertical_dists = []
        for u, l in zip(upper, lower):
            vertical_dists.append(dist.euclidean(u, l))
        avg_vertical = np.mean(vertical_dists)

        # Horizontal distance (corners)
        horizontal = dist.euclidean(corners[0], corners[1])

        return avg_vertical / (horizontal + 1e-6)

    def _smooth_ear(self, raw_ear):
        """Apply exponential moving average to EAR for temporal smoothing."""
        if self.smoothed_ear is None:
            self.smoothed_ear = raw_ear
        else:
            self.smoothed_ear = self.ear_alpha * raw_ear + (1 - self.ear_alpha) * self.smoothed_ear
        return self.smoothed_ear

    def _update_baseline(self, ear):
        """Build adaptive EAR threshold from the user's open-eye baseline."""
        if not self.baseline_established and ear > 0.18:
            self.baseline_ears.append(ear)
            if len(self.baseline_ears) >= 20:   # Faster baseline establishment
                baseline = np.mean(self.baseline_ears)
                # 70% of open-eye EAR is the blink threshold
                self.adaptive_threshold = baseline * 0.70
                self.adaptive_threshold = max(self.adaptive_threshold, 0.13)  # Lower safety floor
                self.adaptive_threshold = min(self.adaptive_threshold, 0.22)  # Safety ceiling
                self.baseline_established = True
                print(f"[INFO] Adaptive EAR baseline: {baseline:.3f}, threshold: {self.adaptive_threshold:.3f}")

    def _check_rapid_drop(self, current_ear):
        """
        Detect a rapid EAR drop relative to recent open-eye average.
        This catches blinks even when smoothed EAR doesn't cross the absolute threshold.
        """
        if len(self.recent_open_ear) < 5:
            return False

        recent_avg = np.mean(self.recent_open_ear)
        drop_pct = (recent_avg - current_ear) / (recent_avg + 1e-6)

        return drop_pct > self.drop_threshold_pct

    def _estimate_head_pose(self, landmarks):
        """
        Estimate head yaw and pitch from facial geometry.
        Uses nose-to-cheek ratios for yaw and nose-to-chin/forehead for pitch.
        """
        nose = np.array(landmarks[NOSE_TIP_INDEX], dtype=np.float64)
        chin = np.array(landmarks[CHIN_INDEX], dtype=np.float64)
        forehead = np.array(landmarks[FOREHEAD_INDEX], dtype=np.float64)
        left_cheek = np.array(landmarks[LEFT_CHEEK_INDEX], dtype=np.float64)
        right_cheek = np.array(landmarks[RIGHT_CHEEK_INDEX], dtype=np.float64)

        # Yaw estimation (left-right head turn)
        left_dist = dist.euclidean(nose, left_cheek)
        right_dist = dist.euclidean(nose, right_cheek)
        yaw = (left_dist - right_dist) / (left_dist + right_dist + 1e-6)

        # Pitch estimation (up-down head tilt)
        chin_dist = dist.euclidean(nose, chin)
        forehead_dist = dist.euclidean(nose, forehead)
        pitch = (forehead_dist - chin_dist) / (forehead_dist + chin_dist + 1e-6)

        self.head_yaw_history.append(yaw)
        self.head_pitch_history.append(pitch)

        return float(yaw), float(pitch)

    def _analyze_micro_movements(self, landmarks):
        """
        Detect if the presentation is completely static (photo/frozen video).
        Real faces always have micro-movements due to breathing and muscle tension.
        """
        key_points = np.array([landmarks[i] for i in [NOSE_TIP_INDEX, CHIN_INDEX,
                               LEFT_CHEEK_INDEX, RIGHT_CHEEK_INDEX,
                               LEFT_TEMPLE_INDEX, RIGHT_TEMPLE_INDEX]], dtype=np.float64)
        self.landmark_history.append(key_points)

        if len(self.landmark_history) >= 8:
            recent = np.array(list(self.landmark_history))
            # Calculate average displacement over recent frames
            diffs = np.diff(recent, axis=0)
            avg_movement = np.mean(np.linalg.norm(diffs, axis=2))

            if avg_movement < self.static_threshold:
                self.static_frame_count += 1
            else:
                self.static_frame_count = max(0, self.static_frame_count - 2)

            return float(avg_movement)
        return 5.0  # Default: assume movement exists

    def _compute_anti_spoof_score(self):
        """
        Compute a composite anti-spoofing score (0.0 = definitely fake, 1.0 = definitely real).
        Combines multiple independent liveness signals.
        """
        score = 0.0
        weights_total = 0.0

        # 1. Blink naturalness (40%)
        w_blink = 0.40
        if self.total_blinks >= self.min_blinks:
            self.spoof_scores['blink_naturalness'] = min(1.0, self.total_blinks / 3.0)
        else:
            self.spoof_scores['blink_naturalness'] = self.total_blinks / (self.min_blinks + 1)
        score += w_blink * self.spoof_scores['blink_naturalness']
        weights_total += w_blink

        # 2. Head dynamism (25%)
        w_head = 0.25
        if len(self.head_yaw_history) >= 5:
            yaw_var = np.var(list(self.head_yaw_history))
            pitch_var = np.var(list(self.head_pitch_history))
            dynamism = min(1.0, (yaw_var + pitch_var) * 100)
            self.spoof_scores['head_dynamism'] = dynamism
        else:
            self.spoof_scores['head_dynamism'] = 0.0
        score += w_head * self.spoof_scores['head_dynamism']
        weights_total += w_head

        # 3. Micro-movement (20%)
        w_micro = 0.20
        if self.static_frame_count > 15:
            self.spoof_scores['micro_movement'] = 0.1  # Very static = suspicious
        elif self.static_frame_count > 5:
            self.spoof_scores['micro_movement'] = 0.5
        else:
            self.spoof_scores['micro_movement'] = 0.9
        score += w_micro * self.spoof_scores['micro_movement']
        weights_total += w_micro

        # 4. EAR variability (15%) — real eyes have natural EAR fluctuation
        w_ear = 0.15
        if len(self.ear_history) >= 10:
            ear_std = np.std(list(self.ear_history)[-20:])
            ear_var_score = min(1.0, ear_std / 0.03)  # Normalize
            self.spoof_scores['ear_variability'] = ear_var_score
        else:
            self.spoof_scores['ear_variability'] = 0.5
        score += w_ear * self.spoof_scores['ear_variability']
        weights_total += w_ear

        return float(score / weights_total) if weights_total > 0 else 0.0

    def get_landmarks(self, frame):
        """
        Extract face landmarks using MediaPipe FaceMesh.

        Args:
            frame (np.ndarray): Full BGR frame.

        Returns:
            list or None: List of (x, y) pixel coordinates for 468 landmarks.
        """
        if self.face_mesh is None:
            return None

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self._lock:
            results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks.landmark
        ]
        return landmarks

    def analyze_frame(self, frame, face_box):
        """
        Analyze a single frame for liveness signals (enhanced v2).

        Uses dual-path blink detection:
          Path 1: Smoothed EAR vs absolute threshold (standard)
          Path 2: Raw EAR vs threshold OR rapid-drop detection (catches subtle blinks)

        Args:
            frame (np.ndarray): Full BGR frame.
            face_box (tuple): (x, y, w, h) from face detector.

        Returns:
            dict: Enhanced liveness analysis results.
        """
        default_result = {
            'ear_left': 0.0, 'ear_right': 0.0, 'ear_avg': 0.0,
            'ear_smoothed': 0.0,
            'eyes_closed': False, 'blink_detected': False,
            'total_blinks': self.total_blinks,
            'head_movement': False, 'liveness_passed': False,
            'head_yaw': 0.0, 'head_pitch': 0.0,
            'micro_movement': 0.0, 'anti_spoof_score': 0.0,
            'spoof_scores': self.spoof_scores.copy(),
            'landmarks': None
        }

        if self.face_mesh is None:
            return default_result

        # ── Get facial landmarks ──
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            return default_result

        # ── Calculate EAR (standard 6-point) ──
        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]

        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0

        # Store raw EAR
        self.raw_ear_history.append(ear_avg)

        # Apply temporal smoothing
        ear_smoothed = self._smooth_ear(ear_avg)
        self.ear_history.append(ear_smoothed)

        # Update adaptive baseline
        self._update_baseline(ear_avg)   # Use raw EAR for baseline

        # Use adaptive threshold if available
        threshold = self.adaptive_threshold if self.baseline_established else self.ear_threshold

        # ═══════════════════════════════════════════════════════════════════
        # DUAL-PATH BLINK DETECTION
        # ═══════════════════════════════════════════════════════════════════

        blink_detected = False

        # ── PATH 1: Smoothed EAR vs threshold (standard method) ──
        eyes_closed_smoothed = ear_smoothed < threshold

        if eyes_closed_smoothed:
            self.blink_counter += 1
        else:
            if (self.blink_counter >= self.min_blink_duration and
                    self.blink_counter <= self.max_blink_duration):
                self.total_blinks += 1
                blink_detected = True
                print(f"[BLINK] Path 1 (smoothed): blink #{self.total_blinks} "
                      f"(EAR={ear_smoothed:.3f}, threshold={threshold:.3f}, "
                      f"duration={self.blink_counter} frames)")
            self.blink_counter = 0

        # ── PATH 2: Raw EAR vs threshold (catches fast blinks) ──
        eyes_closed_raw = ear_avg < threshold

        if eyes_closed_raw:
            self.raw_blink_counter += 1
        else:
            if (self.raw_blink_counter >= 1 and
                    self.raw_blink_counter <= self.max_blink_duration):
                if not blink_detected:  # Don't double-count
                    self.total_blinks += 1
                    blink_detected = True
                    print(f"[BLINK] Path 2 (raw): blink #{self.total_blinks} "
                          f"(raw EAR={ear_avg:.3f}, threshold={threshold:.3f}, "
                          f"duration={self.raw_blink_counter} frames)")
            self.raw_blink_counter = 0

        # ── PATH 3: Rapid-drop detection (relative change) ──
        # Even if EAR doesn't cross absolute threshold, a rapid drop indicates a blink
        if not blink_detected:
            is_rapid_drop = self._check_rapid_drop(ear_avg)
            if is_rapid_drop and not eyes_closed_smoothed and not eyes_closed_raw:
                # Only count if we're seeing a recovery (eyes reopening after drop)
                pass  # We'll track this as part of the drop
            elif not is_rapid_drop and len(self.raw_ear_history) >= 3:
                # Check if previous frames showed a rapid drop that's now recovered
                prev_ears = list(self.raw_ear_history)
                if len(prev_ears) >= 3:
                    recent_avg = np.mean(list(self.recent_open_ear)) if len(self.recent_open_ear) > 0 else ear_avg
                    # Check if any of the last 2 frames had a significant drop
                    for prev_ear in prev_ears[-3:-1]:
                        drop_pct = (recent_avg - prev_ear) / (recent_avg + 1e-6)
                        if drop_pct > self.drop_threshold_pct and ear_avg > prev_ear * 1.15:
                            self.total_blinks += 1
                            blink_detected = True
                            print(f"[BLINK] Path 3 (rapid-drop): blink #{self.total_blinks} "
                                  f"(drop from {recent_avg:.3f} to {prev_ear:.3f}, "
                                  f"recovered to {ear_avg:.3f})")
                            break

        # Track recent open-eye EAR for rapid-drop detection
        eyes_closed = eyes_closed_smoothed or eyes_closed_raw
        if not eyes_closed and ear_avg > 0.18:
            self.recent_open_ear.append(ear_avg)

        # ── Head pose estimation ──
        head_yaw, head_pitch = self._estimate_head_pose(landmarks)

        # ── Head movement tracking ──
        nose_tip = landmarks[NOSE_TIP_INDEX]
        self.nose_positions.append(nose_tip)

        if len(self.nose_positions) >= 5:     # Reduced from 10 for faster detection
            positions = np.array(list(self.nose_positions))
            # Total displacement over recent window
            displacement = np.linalg.norm(
                np.array(positions[-1], dtype=np.float64) - 
                np.array(positions[0], dtype=np.float64)
            )
            # Also check max range of motion
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            total_range = max(x_range, y_range)

            if displacement > self.head_movement_threshold or total_range > self.head_movement_threshold:
                self.head_movement_detected = True

        # ── Micro-movement analysis ──
        micro_movement = self._analyze_micro_movements(landmarks)

        # ── Anti-spoofing score ──
        anti_spoof_score = self._compute_anti_spoof_score()

        # ── Liveness verdict ──
        liveness_passed = self.total_blinks >= self.min_blinks

        return {
            'ear_left': float(ear_left),
            'ear_right': float(ear_right),
            'ear_avg': float(ear_avg),
            'ear_smoothed': float(ear_smoothed),
            'eyes_closed': eyes_closed,
            'blink_detected': blink_detected,
            'total_blinks': self.total_blinks,
            'head_movement': self.head_movement_detected,
            'liveness_passed': liveness_passed,
            'head_yaw': head_yaw,
            'head_pitch': head_pitch,
            'micro_movement': float(micro_movement),
            'anti_spoof_score': float(anti_spoof_score),
            'spoof_scores': self.spoof_scores.copy(),
            'landmarks': landmarks
        }

    def draw_liveness_info(self, frame, result):
        """
        Draw enhanced liveness detection overlay on the frame.

        Args:
            frame (np.ndarray): Frame to annotate.
            result (dict): Result from analyze_frame().

        Returns:
            np.ndarray: Annotated frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Draw eye contours ──
        if result['landmarks'] is not None:
            left_eye_pts = np.array([result['landmarks'][i] for i in LEFT_EYE_INDICES], dtype=np.int32)
            right_eye_pts = np.array([result['landmarks'][i] for i in RIGHT_EYE_INDICES], dtype=np.int32)
            cv2.polylines(annotated, [left_eye_pts], True, (0, 255, 255), 1)
            cv2.polylines(annotated, [right_eye_pts], True, (0, 255, 255), 1)

        # ── Status text ──
        color = (0, 255, 0) if result['liveness_passed'] else (0, 0, 255)
        status = "LIVE" if result['liveness_passed'] else "CHECKING..."
        spoof_pct = result.get('anti_spoof_score', 0) * 100

        cv2.putText(annotated, f"EAR: {result.get('ear_smoothed', 0):.3f} (raw: {result.get('ear_avg', 0):.3f})",
                    (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Blinks: {result['total_blinks']}",
                    (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Head Move: {'Yes' if result['head_movement'] else 'No'}",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Anti-Spoof: {spoof_pct:.0f}%",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Liveness: {status}",
                    (10, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return annotated

    def reset(self):
        """Reset all liveness state for a new authentication session."""
        self.blink_counter = 0
        self.raw_blink_counter = 0
        self.total_blinks = 0
        self.ear_history.clear()
        self.raw_ear_history.clear()
        self.recent_open_ear.clear()
        self.nose_positions.clear()
        self.head_movement_detected = False
        self.smoothed_ear = None
        self.baseline_ears.clear()
        self.baseline_established = False
        self.adaptive_threshold = self.ear_threshold
        self.blink_candidate_start = None
        self.blink_candidate_duration = 0
        self.landmark_history.clear()
        self.static_frame_count = 0
        self.head_yaw_history.clear()
        self.head_pitch_history.clear()
        self.spoof_scores = {k: 0.0 for k in self.spoof_scores}


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from face_detector import FaceDetector

    detector = FaceDetector()
    liveness = LivenessDetector()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    print("[INFO] Starting liveness test. Blink to pass! Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        if faces:
            result = liveness.analyze_frame(frame, faces[0]['box'])
            frame = liveness.draw_liveness_info(frame, result)
            frame = detector.draw_detections(frame, faces)

        cv2.imshow("Liveness Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
