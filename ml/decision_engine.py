"""
=============================================================================
 Module 5: Decision Engine — Risk-Based Multi-Signal Fusion (v2)

 Major upgrade from simple threshold logic to weighted risk scoring:
  - Each module contributes a weighted confidence score
  - Risk penalties applied for suspicious signals
  - Composite risk score determines access level
  - Attack classification with severity grading
  - Temporal consistency validation
=============================================================================
"""

import time
import os
import sys
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DecisionEngine:
    """
    Risk-based multi-signal fusion engine.

    Instead of simple pass/fail gates, each module produces a confidence score
    and the engine computes a composite risk profile:

    Risk Score = 1.0 - Weighted(face_score, recog_score, liveness_score, deepfake_score)

    Score Weights:
      - Face Detection:    10%  (basic prerequisite)
      - Face Recognition:  25%  (identity verification)
      - Liveness Detection: 30%  (anti-spoofing — critical)
      - Deepfake Detection: 35%  (core project focus)

    Decision:
      - Risk < 0.25  → ACCESS GRANTED
      - Risk 0.25-0.50 → SUSPICIOUS — additional verification needed
      - Risk > 0.50  → ACCESS DENIED
    """

    # ── Weight configuration ──
    WEIGHTS = {
        'face_detection':  0.10,
        'face_recognition': 0.25,
        'liveness':         0.30,
        'deepfake':         0.35
    }

    # ── Risk thresholds ──
    RISK_GRANT_THRESHOLD = 0.25     # Below this → GRANTED
    RISK_SUSPICIOUS_THRESHOLD = 0.50  # Below this → SUSPICIOUS (treated as DENIED)

    def __init__(self):
        """Initialize the risk-based decision engine."""
        self.window_size = config.AUTH_WINDOW_FRAMES

        # ── Signal buffers (sliding windows) ──
        self.face_detected_buffer = deque(maxlen=self.window_size)
        self.recognition_buffer = deque(maxlen=self.window_size)
        self.liveness_buffer = deque(maxlen=self.window_size)
        self.deepfake_buffer = deque(maxlen=self.window_size)

        # ── Session state ──
        self.session_start = None
        self.frame_count = 0
        self.final_verdict = None
        self.verdict_reasons = []
        self.is_active = False

        # ── Risk history ──
        self.risk_history = deque(maxlen=30)

    def start_session(self):
        """Start a new authentication session."""
        self.reset()
        self.session_start = datetime.now()
        self.is_active = True

    def reset(self):
        """Clear all buffers and reset state."""
        self.face_detected_buffer.clear()
        self.recognition_buffer.clear()
        self.liveness_buffer.clear()
        self.deepfake_buffer.clear()
        self.frame_count = 0
        self.final_verdict = None
        self.verdict_reasons = []
        self.is_active = False
        self.session_start = None
        self.risk_history.clear()

    def add_signals(self, face_result=None, recognition_result=None,
                    liveness_result=None, deepfake_result=None):
        """
        Add signal data from one frame processing cycle.

        Args:
            face_result (dict): Output from FaceDetector.detect_faces()
            recognition_result (dict): Output from FaceRecognizer.recognize_face()
            liveness_result (dict): Output from LivenessDetector.analyze_frame()
            deepfake_result (dict): Output from DeepfakeDetector.analyze_face()
        """
        self.frame_count += 1

        # ── Face Detection signal ──
        if face_result is not None:
            has_face = len(face_result) > 0 if isinstance(face_result, list) else False
            self.face_detected_buffer.append({
                'detected': has_face,
                'confidence': face_result[0]['confidence'] if has_face else 0.0,
                'frame': self.frame_count
            })

        # ── Face Recognition signal ──
        if recognition_result is not None:
            self.recognition_buffer.append({
                'matched': recognition_result.get('matched', False),
                'username': recognition_result.get('username', 'Unknown'),
                'confidence': recognition_result.get('confidence', 0.0),
                'distance': recognition_result.get('distance', 1.0),
                'frame': self.frame_count
            })

        # ── Liveness signal ──
        if liveness_result is not None:
            self.liveness_buffer.append({
                'ear_avg': liveness_result.get('ear_avg', 0.0),
                'blink_detected': liveness_result.get('blink_detected', False),
                'total_blinks': liveness_result.get('total_blinks', 0),
                'head_movement': liveness_result.get('head_movement', False),
                'liveness_passed': liveness_result.get('liveness_passed', False),
                'anti_spoof_score': liveness_result.get('anti_spoof_score', 0.0),
                'frame': self.frame_count
            })

        # ── Deepfake signal ──
        if deepfake_result is not None:
            self.deepfake_buffer.append({
                'is_real': deepfake_result.get('is_real', False),
                'confidence_real': deepfake_result.get('confidence_real', 0.0),
                'label': deepfake_result.get('label', 'UNKNOWN'),
                'frame': self.frame_count
            })

    def _compute_face_score(self):
        """Compute face detection confidence score (0-1)."""
        if len(self.face_detected_buffer) == 0:
            return 0.0, False, "No face detection data."

        recent = list(self.face_detected_buffer)
        rate = sum(1 for f in recent if f['detected']) / len(recent)
        avg_conf = sum(f['confidence'] for f in recent) / len(recent)

        score = rate * avg_conf
        passed = rate > 0.5
        reason = None if passed else "No consistent face detected in frame."
        return score, passed, reason

    def _compute_recognition_score(self):
        """Compute face recognition confidence score (0-1)."""
        if len(self.recognition_buffer) == 0:
            return 0.0, False, "No recognition data yet.", "Unknown"

        recent = list(self.recognition_buffer)

        # Require at least 3 recognition samples before declaring a match
        # to prevent premature acceptance on noisy early frames
        if len(recent) < 3:
            return 0.0, False, "Collecting recognition data...", "Unknown"

        matches = [r for r in recent if r['matched']]
        match_rate = len(matches) / len(recent)

        if matches:
            usernames = [m['username'] for m in matches]
            username = max(set(usernames), key=usernames.count)
            avg_conf = sum(m['confidence'] for m in matches) / len(matches)
            score = match_rate * avg_conf
        else:
            username = "Unknown"
            score = 0.0

        # Require 60% match rate (stricter than 50%) to reduce false acceptance
        passed = match_rate > 0.6
        reason = None if passed else "Face does not match any registered user."
        return score, passed, reason, username

    def _compute_liveness_score(self):
        """
        Compute liveness score (0-1) using both classic blink detection
        and the enhanced anti-spoofing score from the liveness detector.
        """
        if len(self.liveness_buffer) == 0:
            return 0.0, False, "No liveness data yet."

        recent = list(self.liveness_buffer)
        latest = recent[-1]
        total_blinks = latest['total_blinks']
        anti_spoof = latest.get('anti_spoof_score', 0.0)
        head_movement = any(l['head_movement'] for l in recent)
        liveness_passed = latest['liveness_passed']

        # Composite liveness score
        blink_score = min(1.0, total_blinks / max(self.min_blinks_required(), 1))
        head_score = 0.8 if head_movement else 0.2

        # Blend: 50% blink + 25% anti-spoof + 25% head
        score = 0.50 * blink_score + 0.25 * anti_spoof + 0.25 * head_score

        passed = liveness_passed
        reason = None
        if not passed:
            reason = f"Liveness check failed — blinks: {total_blinks} (need {config.MIN_BLINKS_REQUIRED})."
        return score, passed, reason

    def min_blinks_required(self):
        """Return minimum blinks required."""
        return config.MIN_BLINKS_REQUIRED

    def _compute_deepfake_score(self):
        """
        Compute deepfake detection score (0-1).
        Higher score = more likely real.
        """
        if len(self.deepfake_buffer) == 0:
            return 0.0, False, "No deepfake analysis data yet."

        recent = list(self.deepfake_buffer)
        real_confs = [d['confidence_real'] for d in recent]
        avg_real_conf = sum(real_confs) / len(real_confs)
        fake_count = sum(1 for d in recent if not d['is_real'])
        fake_rate = fake_count / len(recent)

        # Apply fake rate penalty (gentle — heuristics naturally oscillate frame-to-frame)
        score = avg_real_conf * (1.0 - fake_rate * 0.15)

        passed = avg_real_conf >= config.DEEPFAKE_REAL_THRESHOLD
        reason = None
        if avg_real_conf < config.DEEPFAKE_SUSPICIOUS_THRESHOLD:
            reason = f"DEEPFAKE DETECTED — confidence: {avg_real_conf:.2f}"
        elif avg_real_conf < config.DEEPFAKE_REAL_THRESHOLD:
            reason = f"Suspicious face detected — confidence: {avg_real_conf:.2f}"

        return score, passed, reason

    def _classify_attack(self, scores, reasons):
        """
        Classify the type and severity of detected attack.

        Returns:
            tuple: (alert: bool, alert_type: str, severity: str)
        """
        alert = False
        alert_type = None
        severity = "NONE"

        # Check deepfake attack (raised thresholds — only flag truly extreme scores)
        df_score = scores.get('deepfake', {}).get('score', 1.0)
        if df_score < 0.30:
            alert = True
            alert_type = "DEEPFAKE_DETECTED"
            severity = "CRITICAL"
        elif df_score < 0.50:
            alert = True
            alert_type = "SUSPICIOUS_FACE"
            severity = "HIGH"

        # Check liveness attack (photo/video replay)
        live_score = scores.get('liveness', {}).get('score', 1.0)
        blinks = scores.get('liveness', {}).get('total_blinks', 0)
        if blinks == 0 and self.frame_count > 30:
            if not alert or severity != "CRITICAL":
                alert = True
                alert_type = "POSSIBLE_PHOTO_ATTACK"
                severity = "HIGH"

        # Check for temporal inconsistency (adversarial)
        if len(self.risk_history) >= 10:
            risk_vals = list(self.risk_history)[-10:]
            risk_variance = max(risk_vals) - min(risk_vals)
            if risk_variance > 0.5:  # High variability = suspicious
                alert = True
                if alert_type is None:
                    alert_type = "ADVERSARIAL_ANOMALY"
                    severity = "MEDIUM"

        return alert, alert_type, severity

    def evaluate(self):
        """
        Evaluate all accumulated signals using risk-based scoring.

        Returns:
            dict: Complete verdict with risk analysis.
        """
        reasons = []
        scores = {}
        username = "Unknown"

        # ── 1. Face Detection Score ──
        face_score, face_passed, face_reason = self._compute_face_score()
        scores['face_detection'] = {
            'score': face_score,
            'passed': face_passed,
            'weight': self.WEIGHTS['face_detection']
        }
        if face_reason:
            reasons.append(face_reason)

        # ── 2. Face Recognition Score ──
        recog_score, recog_passed, recog_reason, username = self._compute_recognition_score()
        scores['face_recognition'] = {
            'score': recog_score,
            'passed': recog_passed,
            'weight': self.WEIGHTS['face_recognition'],
            'username': username
        }
        if recog_reason:
            reasons.append(recog_reason)

        # ── 3. Liveness Score ──
        live_score, live_passed, live_reason = self._compute_liveness_score()
        if len(self.liveness_buffer) > 0:
            latest = list(self.liveness_buffer)[-1]
            scores['liveness'] = {
                'score': live_score,
                'passed': live_passed,
                'weight': self.WEIGHTS['liveness'],
                'total_blinks': latest['total_blinks'],
                'head_movement': any(l['head_movement'] for l in self.liveness_buffer),
                'anti_spoof_score': latest.get('anti_spoof_score', 0.0)
            }
        else:
            scores['liveness'] = {
                'score': 0.0, 'passed': False,
                'weight': self.WEIGHTS['liveness'],
                'total_blinks': 0, 'head_movement': False,
                'anti_spoof_score': 0.0
            }
        if live_reason:
            reasons.append(live_reason)

        # ── 4. Deepfake Score ──
        df_score, df_passed, df_reason = self._compute_deepfake_score()
        if len(self.deepfake_buffer) > 0:
            recent_df = list(self.deepfake_buffer)
            scores['deepfake'] = {
                'score': df_score,
                'passed': df_passed,
                'weight': self.WEIGHTS['deepfake'],
                'avg_real_confidence': sum(d['confidence_real'] for d in recent_df) / len(recent_df),
                'fake_rate': sum(1 for d in recent_df if not d['is_real']) / len(recent_df)
            }
        else:
            scores['deepfake'] = {
                'score': 0.0, 'passed': False,
                'weight': self.WEIGHTS['deepfake'],
                'avg_real_confidence': 0, 'fake_rate': 0
            }
        if df_reason:
            reasons.append(df_reason)

        # ── Compute Composite Risk Score ──
        composite_confidence = (
            self.WEIGHTS['face_detection'] * face_score +
            self.WEIGHTS['face_recognition'] * recog_score +
            self.WEIGHTS['liveness'] * live_score +
            self.WEIGHTS['deepfake'] * df_score
        )
        risk_score = 1.0 - composite_confidence
        self.risk_history.append(risk_score)

        # ── Attack Classification ──
        alert, alert_type, severity = self._classify_attack(scores, reasons)

        # ── Final Decision ──
        min_frames = max(config.DEEPFAKE_EVERY_N, config.FACE_RECOG_EVERY_N) * 2
        has_enough_data = self.frame_count >= min_frames

        if has_enough_data:
            all_passed = all([face_passed, recog_passed, live_passed, df_passed])

            if all_passed and risk_score < self.RISK_GRANT_THRESHOLD:
                decision = "GRANTED"
                reasons = ["All authentication checks passed."]
            elif risk_score < self.RISK_SUSPICIOUS_THRESHOLD and all_passed:
                decision = "GRANTED"
                reasons = ["Authentication passed with elevated monitoring."]
            else:
                decision = "DENIED"
        else:
            decision = "PENDING"
            if not reasons:
                reasons.append("Collecting more data...")

        # Calculate session duration
        duration = 0.0
        if self.session_start:
            duration = (datetime.now() - self.session_start).total_seconds()

        self.final_verdict = {
            'decision': decision,
            'reasons': reasons,
            'scores': scores,
            'risk_score': round(risk_score, 4),
            'composite_confidence': round(composite_confidence, 4),
            'username': username,
            'alert': alert,
            'alert_type': alert_type,
            'alert_severity': severity,
            'frame_count': self.frame_count,
            'session_duration': duration
        }

        return self.final_verdict

    def get_status_summary(self):
        """
        Get a compact status summary for display.

        Returns:
            dict: Simplified status for UI rendering.
        """
        verdict = self.evaluate()
        return {
            'decision': verdict['decision'],
            'username': verdict['username'],
            'alert': verdict['alert'],
            'alert_type': verdict['alert_type'],
            'alert_severity': verdict.get('alert_severity', 'NONE'),
            'risk_score': verdict.get('risk_score', 1.0),
            'composite_confidence': verdict.get('composite_confidence', 0.0),
            'face_ok': verdict['scores'].get('face_detection', {}).get('passed', False),
            'recog_ok': verdict['scores'].get('face_recognition', {}).get('passed', False),
            'live_ok': verdict['scores'].get('liveness', {}).get('passed', False),
            'deepfake_ok': verdict['scores'].get('deepfake', {}).get('passed', False),
            'blinks': verdict['scores'].get('liveness', {}).get('total_blinks', 0),
            'reasons': verdict['reasons']
        }
