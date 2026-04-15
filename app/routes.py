"""
=============================================================================
 Flask Routes — HTTP endpoints for Registration, Login, and Dashboard.
=============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user

import config
from utils.db_utils import (
    create_user, get_user_by_username, verify_password,
    get_login_history, get_all_users, log_audit
)
from app import login_manager, User

# ── Blueprint ──────────────────────────────────────────────────────────────
main_bp = Blueprint('main', __name__)


# ── Landing Page ───────────────────────────────────────────────────────────
@main_bp.route('/')
def index():
    """Landing page with system overview."""
    return render_template('index.html')


# ── Registration ───────────────────────────────────────────────────────────
@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page — capture face + create account."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')

        # Check if user exists
        if get_user_by_username(username):
            flash('Username already taken.', 'error')
            return render_template('register.html')

        # User will be fully created after face capture via SocketIO
        # Store pending registration in session
        session['pending_registration'] = {
            'username': username,
            'email': email,
            'password': password
        }

        return render_template('register.html', capture_mode=True,
                             username=username, email=email)

    return render_template('register.html')


# ── Login ──────────────────────────────────────────────────────────────────
@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page — password + face authentication."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('login.html')

        # Verify password first
        if not verify_password(username, password):
            flash('Invalid username or password.', 'error')
            log_audit("LOGIN_FAILED", username, "Invalid password attempt.")
            return render_template('login.html')

        # Password OK — proceed to face authentication
        session['auth_username'] = username
        return render_template('login.html', face_auth_mode=True, username=username)

    return render_template('login.html')


# ── Dashboard ──────────────────────────────────────────────────────────────
@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Post-login dashboard with user info and login history."""
    history = get_login_history(current_user.username, limit=20)
    users = get_all_users()
    return render_template('dashboard.html',
                         user=current_user,
                         history=history,
                         users=users)


# ── Logout ─────────────────────────────────────────────────────────────────
@main_bp.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    log_audit("USER_LOGOUT", current_user.username, "User logged out.")
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))


# ── API: Complete Registration ─────────────────────────────────────────────
@main_bp.route('/api/complete-registration', methods=['POST'])
def complete_registration():
    """
    API endpoint called after face capture is complete.
    Creates the user in the database.
    """
    pending = session.get('pending_registration')
    if not pending:
        return jsonify({'success': False, 'message': 'No pending registration.'}), 400

    data = request.get_json()
    num_encodings = data.get('num_encodings', 0)

    result = create_user(
        pending['username'],
        pending['email'],
        pending['password'],
        num_encodings=num_encodings
    )

    if result['success']:
        session.pop('pending_registration', None)

    return jsonify(result)


# ── API: Complete Face Auth ────────────────────────────────────────────────
@main_bp.route('/api/complete-face-auth', methods=['POST'])
def complete_face_auth():
    """
    API endpoint called after face authentication succeeds.
    Logs in the user via Flask-Login.
    """
    username = session.get('auth_username')
    if not username:
        return jsonify({'success': False, 'message': 'No auth session.'}), 400

    user_data = get_user_by_username(username)
    if not user_data:
        return jsonify({'success': False, 'message': 'User not found.'}), 404

    user = User(user_data)
    login_user(user)
    session.pop('auth_username', None)

    log_audit("LOGIN_SUCCESS", username, "Face authentication successful.")

    return jsonify({'success': True, 'redirect': url_for('main.dashboard')})


# ── API: Login History ─────────────────────────────────────────────────────
@main_bp.route('/api/history')
@login_required
def api_history():
    """Get login history for the current user."""
    history = get_login_history(current_user.username, limit=50)
    return jsonify(history)


# ── API: Attack History ────────────────────────────────────────────────────
@main_bp.route('/api/attack-history')
@login_required
def api_attack_history():
    """Get attack/alert history with details."""
    history = get_login_history(limit=100)
    attacks = [h for h in history if h.get('alert_type')]
    return jsonify(attacks)


# ── API: System Stats ─────────────────────────────────────────────────────
@main_bp.route('/api/system-stats')
@login_required
def api_system_stats():
    """Get aggregated system statistics for dashboard charts."""
    history = get_login_history(limit=500)
    users = get_all_users()

    total = len(history)
    success = sum(1 for h in history if h.get('status') == 'SUCCESS')
    denied = sum(1 for h in history if h.get('status') == 'DENIED')
    alerts = sum(1 for h in history if h.get('alert_type'))

    # Attack type breakdown
    attack_types = {}
    for h in history:
        at = h.get('alert_type')
        if at:
            attack_types[at] = attack_types.get(at, 0) + 1

    # Average confidence scores from successful logins
    face_confs = [h['face_confidence'] for h in history
                  if h.get('face_confidence') and h.get('status') == 'SUCCESS']
    df_confs = [h['deepfake_confidence'] for h in history
                if h.get('deepfake_confidence') and h.get('status') == 'SUCCESS']

    return jsonify({
        'total_attempts': total,
        'successful': success,
        'denied': denied,
        'alerts': alerts,
        'registered_users': len(users),
        'attack_types': attack_types,
        'avg_face_confidence': round(sum(face_confs) / len(face_confs), 3) if face_confs else 0,
        'avg_deepfake_confidence': round(sum(df_confs) / len(df_confs), 3) if df_confs else 0,
        'success_rate': round(success / total * 100, 1) if total > 0 else 0
    })


# ── API: Audit Logs ───────────────────────────────────────────────────────
@main_bp.route('/api/audit-logs')
@login_required
def api_audit_logs():
    """Get recent audit log entries."""
    from utils.db_utils import get_audit_logs
    logs = get_audit_logs(limit=50)
    return jsonify(logs)


# ── Dataset & Model Page ──────────────────────────────────────────────────
@main_bp.route('/dataset')
def dataset():
    """Dataset info, model architecture, and image upload test page."""
    import json

    # Load training metrics if available
    training_metrics = None
    metrics_path = os.path.join(config.TRAINING_RESULTS_DIR, 'training_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                training_metrics = json.load(f)
            # Convert dict to object-like access for Jinja2
            class MetricsObj:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
                def get(self, key, default=None):
                    return getattr(self, key, default)
            training_metrics = MetricsObj(training_metrics)
        except Exception:
            training_metrics = None

    # Check if training curves exist
    curves_path = os.path.join(config.TRAINING_RESULTS_DIR, 'training_curves.png')
    training_curves_exist = os.path.exists(curves_path)

    return render_template('dataset.html',
                         training_metrics=training_metrics,
                         training_curves_exist=training_curves_exist)


# ── Training Curves Image ────────────────────────────────────────────────
@main_bp.route('/api/training-curves.png')
def training_curves_image():
    """Serve the training curves image."""
    from flask import send_file
    curves_path = os.path.join(config.TRAINING_RESULTS_DIR, 'training_curves.png')
    if os.path.exists(curves_path):
        return send_file(curves_path, mimetype='image/png')
    return '', 404


# ── API: Predict Image (Deepfake Test with Grad-CAM) ─────────────────────
@main_bp.route('/api/predict-image', methods=['POST'])
def api_predict_image():
    """
    Upload a face image and get deepfake prediction with Grad-CAM heatmap.
    Used by the Dataset & Model page for live testing.
    """
    import cv2
    import numpy as np

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        # Read image file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image.'}), 400

        # Resize if very large
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Initialize detector (lazy load)
        from ml.deepfake_detector import DeepfakeDetector
        if not hasattr(api_predict_image, '_detector'):
            api_predict_image._detector = DeepfakeDetector()

        detector = api_predict_image._detector

        # Get prediction with Grad-CAM
        result = detector.predict_single_image(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Deepfake Analyzer Page ────────────────────────────────────────────────
@main_bp.route('/analyze')
def analyze():
    """Deepfake analysis tool — upload image or video for detection."""
    return render_template('analyze.html')


# ── API: Predict Video (Frame-by-Frame Deepfake Analysis) ─────────────────
@main_bp.route('/api/predict-video', methods=['POST'])
def api_predict_video():
    """
    Upload a video and get per-frame deepfake analysis.
    Extracts frames at regular intervals and runs the detection pipeline.
    """
    import cv2
    import numpy as np
    import tempfile

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        # Save video to a temp file for OpenCV to read
        temp_dir = os.path.join(config.BASE_DIR, 'data', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, 'upload_video.mp4')
        file.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file.'}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Sample up to 10 frames evenly distributed
        max_samples = 10
        step = max(total_frames // max_samples, 1)

        # Initialize detector
        from ml.deepfake_detector import DeepfakeDetector
        if not hasattr(api_predict_video, '_detector'):
            api_predict_video._detector = DeepfakeDetector()
        detector = api_predict_video._detector

        frame_results = []
        frame_idx = 0
        sample_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0 and sample_num < max_samples:
                # Resize if large
                h, w = frame.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                # Use the global detector cache from api_predict_image to save memory
                from ml.face_detector import FaceDetector
                if not hasattr(api_predict_video, '_face_detector'):
                    api_predict_video._face_detector = FaceDetector()
                
                faces = api_predict_video._face_detector.detect_faces(frame)
                
                if len(faces) == 0:
                    frame_results.append({
                        'frame_number': frame_idx + 1,
                        'timestamp': round(frame_idx / fps, 2),
                        'is_real': False,
                        'confidence_real': 0.0,
                        'label': 'NO_FACE'
                    })
                else:
                    face_crop = faces[0]['roi']
                    result = detector.analyze_face(face_crop)
                    frame_results.append({
                        'frame_number': frame_idx + 1,
                        'timestamp': round(frame_idx / fps, 2),
                        'is_real': result.get('is_real', False),
                        'confidence_real': round(result.get('confidence_real', 0.0), 4),
                        'label': result.get('label', 'UNKNOWN')
                    })
                    
                sample_num += 1

            frame_idx += 1

        cap.release()

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        if not frame_results:
            return jsonify({'error': 'No frames could be analyzed.'}), 400

        # Compute summary
        avg_conf = sum(f['confidence_real'] for f in frame_results) / len(frame_results)
        fake_count = sum(1 for f in frame_results if not f['is_real'])

        summary = {
            'is_real': avg_conf >= config.DEEPFAKE_REAL_THRESHOLD,
            'confidence_real': round(avg_conf, 4),
            'label': 'REAL' if avg_conf >= config.DEEPFAKE_REAL_THRESHOLD else (
                'SUSPICIOUS' if avg_conf >= config.DEEPFAKE_SUSPICIOUS_THRESHOLD else 'FAKE'),
            'frames_analyzed': len(frame_results),
            'fake_frames': fake_count,
            'total_frames': total_frames
        }

        return jsonify({
            'summary': summary,
            'frames': frame_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
