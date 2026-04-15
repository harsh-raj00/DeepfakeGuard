"""
=============================================================================
 SocketIO Event Handlers — Real-time webcam frame processing.
 Handles both registration (face capture) and login (authentication) flows.
=============================================================================
"""

import os
import sys
import cv2
import numpy as np
import base64
import time
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_socketio import emit
from app import socketio
import config
from utils.image_utils import base64_to_frame, frame_to_base64
from utils.db_utils import log_login_attempt, log_audit
from utils.logger import logger

# ── Global ML module instances (lazy-loaded) ───────────────────────────────
_face_detector = None
_face_recognizer = None
_liveness_detector = None
_deepfake_detector = None
_init_lock = Lock()
_init_done = False


def get_ml_modules():
    """Lazy-load ML modules to avoid slow startup."""
    global _face_detector, _face_recognizer, _liveness_detector, _deepfake_detector, _init_done

    if _init_done:
        return _face_detector, _face_recognizer, _liveness_detector, _deepfake_detector

    with _init_lock:
        if _init_done:
            return _face_detector, _face_recognizer, _liveness_detector, _deepfake_detector

        if _face_detector is None:
            try:
                from ml.face_detector import FaceDetector
                _face_detector = FaceDetector()
                logger.info("FaceDetector loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load FaceDetector: {e}")

        if _face_recognizer is None:
            try:
                from ml.face_recognizer import FaceRecognizer
                _face_recognizer = FaceRecognizer()
                logger.info("FaceRecognizer loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load FaceRecognizer: {e}")

        if _liveness_detector is None:
            try:
                from ml.liveness_detector import LivenessDetector
                _liveness_detector = LivenessDetector()
                logger.info("LivenessDetector loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load LivenessDetector: {e}")

        if _deepfake_detector is None:
            try:
                from ml.deepfake_detector import DeepfakeDetector
                _deepfake_detector = DeepfakeDetector()
                logger.info("DeepfakeDetector loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load DeepfakeDetector: {e}")

        _init_done = True

    return _face_detector, _face_recognizer, _liveness_detector, _deepfake_detector


# ── Session State Storage ──────────────────────────────────────────────────
# Per-client state (keyed by custom session ID sent from client)
client_state = {}
# Lock to prevent concurrent frame processing (OpenCV DNN is not thread-safe)
_frame_lock = Lock()


# ══════════════════════════════════════════════════════════════════════════
#  CONNECTION EVENTS
# ══════════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def handle_connect(auth=None):
    """Handle new WebSocket connection."""
    from flask import request
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to authentication server.'})


@socketio.on('disconnect')
def handle_disconnect(reason=None):
    """Log disconnect. We intentionally do NOT clear client_state here
    because the client-generated session_id stays the same across
    SocketIO reconnects (transport close → reopen). Clearing state
    would cause all in-flight registration/auth frames to be lost."""
    from flask import request
    logger.info(f"Client disconnected: {request.sid} (reason: {reason})")


# ══════════════════════════════════════════════════════════════════════════
#  REGISTRATION — FACE CAPTURE
# ══════════════════════════════════════════════════════════════════════════

@socketio.on('start_registration')
def handle_start_registration(data):
    """
    Start the face capture process for registration.
    Client sends username, we capture N face frames.
    """
    username = data.get('username', '')
    sid = data.get('sid', 'default')

    logger.info(f"Starting registration capture for: {username} (sid: {sid})")

    client_state[sid] = {
        'mode': 'registration',
        'username': username,
        'captured_faces': [],
        'frame_count': 0,
        'required_frames': config.REGISTRATION_MIN_FRAMES
    }

    emit('registration_status', {
        'status': 'capturing',
        'message': f'Look at the camera. Capturing face (0/{config.REGISTRATION_MIN_FRAMES})...',
        'captured': 0,
        'required': config.REGISTRATION_MIN_FRAMES
    })


@socketio.on('registration_frame')
def handle_registration_frame(data):
    """Process a frame during registration (face capture)."""
    # Drop frame if still processing a previous one (prevents queue buildup)
    if not _frame_lock.acquire(blocking=False):
        return
    try:
        _handle_registration_frame_inner(data)
    finally:
        _frame_lock.release()


def _handle_registration_frame_inner(data):
    """Actual registration frame processing (called under lock)."""
    sid = data.get('sid', 'default')
    frame_data = data.get('frame', '')

    if sid not in client_state or client_state[sid]['mode'] != 'registration':
        logger.warning(f"registration_frame from unknown sid: {sid}")
        return

    state = client_state[sid]
    face_detector, face_recognizer, _, _ = get_ml_modules()

    if face_detector is None:
        emit('registration_status', {
            'status': 'error',
            'message': 'Face detection module not available. Check server logs.'
        })
        return

    # Decode frame
    try:
        frame = base64_to_frame(frame_data)
        if frame is None:
            logger.warning("Could not decode frame")
            return
    except Exception as e:
        logger.warning(f"Frame decode error: {e}")
        return

    state['frame_count'] += 1
    logger.debug(f"[Registration] Frame {state['frame_count']} received and decoded. Shape: {frame.shape}")

    # Detect faces
    faces = face_detector.detect_faces(frame)
    logger.debug(f"[Registration] Frame {state['frame_count']}: {len(faces)} face(s) detected")

    if len(faces) == 0:
        # No face detected — send annotated frame back
        annotated = frame_to_base64(frame)
        emit('processed_frame', {
            'frame': annotated,
            'status': 'No face detected — adjust position.'
        })
        return

    if len(faces) > 1:
        # Multiple faces — warn
        annotated_frame = face_detector.draw_detections(frame, faces, color=(0, 0, 255))
        annotated = frame_to_base64(annotated_frame)
        emit('processed_frame', {
            'frame': annotated,
            'status': 'Multiple faces detected — only one person please.'
        })
        return

    # Single face detected — capture it
    face = faces[0]
    face_roi = face['roi']

    # Save face crop for encoding (only if face is big enough)
    if face_roi.shape[0] > 50 and face_roi.shape[1] > 50:
        # Only capture every 3rd frame to get some variety
        if state['frame_count'] % 3 == 0:
            state['captured_faces'].append(face_roi.copy())

    # Draw detection on frame
    captured_count = len(state['captured_faces'])
    annotated_frame = face_detector.draw_detections(
        frame, faces,
        label=f"Capturing... ({captured_count}/{state['required_frames']})",
        color=(0, 255, 0)
    )
    annotated = frame_to_base64(annotated_frame)

    emit('processed_frame', {'frame': annotated, 'status': 'Face detected!'})
    emit('registration_status', {
        'status': 'capturing',
        'message': f"Captured {captured_count}/{state['required_frames']} frames.",
        'captured': captured_count,
        'required': state['required_frames']
    })

    # Check if we have enough frames
    if captured_count >= state['required_frames']:
        logger.info(f"Registration: captured {captured_count} frames for {state['username']}")

        # Generate encodings and register
        if face_recognizer is not None:
            result = face_recognizer.register_user(
                state['username'],
                state['captured_faces']
            )
            emit('registration_status', {
                'status': 'complete' if result['success'] else 'error',
                'message': result['message'],
                'num_encodings': result.get('num_encodings', 0)
            })
        else:
            emit('registration_status', {
                'status': 'complete',
                'message': 'Face captured successfully (recognition module in fallback mode).',
                'num_encodings': captured_count
            })

        # Clean up
        if sid in client_state:
            del client_state[sid]


# ══════════════════════════════════════════════════════════════════════════
#  LOGIN — FACE AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════

@socketio.on('start_auth')
def handle_start_auth(data):
    """Start the face authentication process."""
    username = data.get('username', '')
    sid = data.get('sid', 'default')

    logger.info(f"Starting face authentication for: {username} (sid: {sid})")

    # Initialize decision engine for this session
    from ml.decision_engine import DecisionEngine
    engine = DecisionEngine()
    engine.start_session()

    client_state[sid] = {
        'mode': 'authentication',
        'username': username,
        'engine': engine,
        'frame_count': 0,
        'start_time': time.time()
    }

    # Reset liveness detector for new session
    _, _, liveness_detector, _ = get_ml_modules()
    if liveness_detector:
        liveness_detector.reset()

    emit('auth_status', {
        'status': 'started',
        'message': 'Look at the camera and blink naturally.',
        'username': username,
        'decision': 'PENDING',
        'face_ok': False,
        'recog_ok': False,
        'live_ok': False,
        'deepfake_ok': False,
        'blinks': 0,
        'alert': False,
        'alert_type': None,
        'reasons': ['Starting authentication...'],
        'frame_count': 0
    })


@socketio.on('auth_frame')
def handle_auth_frame(data):
    """Process a frame during face authentication."""
    # Drop frame if still processing a previous one
    if not _frame_lock.acquire(blocking=False):
        return
    try:
        _handle_auth_frame_inner(data)
    finally:
        _frame_lock.release()


def _handle_auth_frame_inner(data):
    """Actual auth frame processing (called under lock)."""
    sid = data.get('sid', 'default')
    frame_data = data.get('frame', '')

    if sid not in client_state or client_state[sid]['mode'] != 'authentication':
        return

    state = client_state[sid]
    username = state['username']
    engine = state['engine']
    state['frame_count'] += 1
    frame_num = state['frame_count']

    face_detector, face_recognizer, liveness_detector, deepfake_detector = get_ml_modules()

    if face_detector is None:
        emit('auth_status', {
            'status': 'error', 'message': 'ML modules not available.',
            'decision': 'DENIED', 'face_ok': False, 'recog_ok': False,
            'live_ok': False, 'deepfake_ok': False, 'blinks': 0,
            'alert': False, 'alert_type': None, 'reasons': ['ML modules unavailable'],
            'frame_count': frame_num
        })
        return

    # Decode frame
    try:
        frame = base64_to_frame(frame_data)
        if frame is None:
            return
    except Exception:
        return

    # ── 1. Face Detection (every frame) ──
    faces = face_detector.detect_faces(frame)
    engine.add_signals(face_result=faces)

    if len(faces) == 0:
        annotated = frame_to_base64(frame)
        emit('processed_frame', {'frame': annotated, 'status': 'No face detected'})
        # Still send status so UI updates
        verdict = engine.get_status_summary()
        verdict['frame_count'] = frame_num
        verdict['status'] = 'pending'
        emit('auth_status', verdict)
        return

    face = faces[0]
    face_roi = face['roi']
    face_box = face['box']
    annotated_frame = frame.copy()

    # ── 2. Face Recognition (every Nth frame) ──
    if frame_num % config.FACE_RECOG_EVERY_N == 0 and face_recognizer is not None:
        recog_result = face_recognizer.verify_user(username, face_roi)
        engine.add_signals(recognition_result=recog_result)

    # ── 3. Liveness Detection (every frame) ──
    if liveness_detector is not None:
        liveness_result = liveness_detector.analyze_frame(frame, face_box)
        engine.add_signals(liveness_result=liveness_result)
        annotated_frame = liveness_detector.draw_liveness_info(annotated_frame, liveness_result)

    # ── 4. Deepfake Detection (every Nth frame) ──
    if frame_num % config.DEEPFAKE_EVERY_N == 0 and deepfake_detector is not None:
        deepfake_result = deepfake_detector.analyze_face(face_roi)
        engine.add_signals(deepfake_result=deepfake_result)

    # ── Evaluate Decision Engine ──
    verdict = engine.get_status_summary()

    # ── Draw Face Detection ──
    det_color = (0, 255, 0) if verdict['recog_ok'] else (0, 165, 255)
    annotated_frame = face_detector.draw_detections(
        annotated_frame, faces,
        label=username if verdict['recog_ok'] else "Verifying...",
        color=det_color
    )

    # ── Draw deepfake/alert overlays ──
    if verdict['alert']:
        h, w = annotated_frame.shape[:2]
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        alert_text = f"ALERT: {verdict['alert_type']}"
        cv2.putText(annotated_frame, alert_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Send annotated frame back
    b64_frame = frame_to_base64(annotated_frame)
    emit('processed_frame', {
        'frame': b64_frame,
        'status': verdict['decision']
    })

    # Send detailed status
    verdict['frame_count'] = frame_num
    verdict['status'] = verdict['decision'].lower()
    emit('auth_status', verdict)

    # ── Handle final decision ──
    if verdict['decision'] == 'GRANTED':
        # Authentication successful!
        full_eval = engine.evaluate()
        log_login_attempt(
            username, 'SUCCESS',
            face_confidence=full_eval['scores'].get('face_recognition', {}).get('avg_confidence', 0),
            liveness_blinks=verdict['blinks'],
            deepfake_confidence=full_eval['scores'].get('deepfake', {}).get('avg_real_confidence', 0),
            details="All checks passed."
        )
        emit('auth_complete', {
            'status': 'granted',
            'message': f'Welcome, {username}! Authentication successful.',
            'username': username
        })
        # Clean up
        if sid in client_state:
            del client_state[sid]
        return

    # Timeout after 45 seconds (extended for reliable blink detection at 8 FPS)
    elapsed = time.time() - state['start_time']
    if elapsed > 45:
        log_login_attempt(
            username, 'DENIED',
            alert_type=verdict.get('alert_type'),
            details='; '.join(verdict['reasons'])
        )
        emit('auth_complete', {
            'status': 'denied',
            'message': 'Authentication timed out. Please try again.',
            'reasons': verdict['reasons']
        })
        if sid in client_state:
            del client_state[sid]
