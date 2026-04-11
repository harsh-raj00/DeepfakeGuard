/**
 * =============================================================================
 * FaceAuth Guard — Client-Side Webcam & SocketIO Logic
 * Handles webcam capture, frame streaming, and real-time UI updates
 * for both Registration (face capture) and Login (face authentication).
 * =============================================================================
 */

// ── Global State ────────────────────────────────────────────────────────────
let socket = null;
let videoStream = null;
let webcamVideo = null;
let canvas = null;
let ctx = null;
let processedFeed = null;
let isStreaming = false;
let frameInterval = null;
const FRAME_RATE = 8; // Frames per second to send to server
const SESSION_ID = 'session_' + Math.random().toString(36).substr(2, 9);

// ── Initialize SocketIO Connection ──────────────────────────────────────────
function initSocket() {
    if (socket && socket.connected) {
        console.log('[SocketIO] Already connected.');
        return;
    }

    console.log('[SocketIO] Attempting to connect...');

    // Connect to the same host the page was loaded from
    socket = io(window.location.origin, {
        transports: ['polling', 'websocket'],  // Start with polling (more reliable), upgrade to ws
        reconnection: true,
        reconnectionAttempts: 10,
        reconnectionDelay: 1000,
        timeout: 10000
    });

    socket.on('connect', () => {
        console.log('[SocketIO] Connected! ID:', socket.id);
        updateStatus('Connected to server.', 'info');
        updateCameraStatus('Camera active — connected');
    });

    socket.on('connect_error', (err) => {
        console.error('[SocketIO] Connection error:', err.message);
        updateStatus('Connection error: ' + err.message, 'error');
    });

    socket.on('disconnect', (reason) => {
        console.log('[SocketIO] Disconnected:', reason);
        updateStatus('Disconnected from server.', 'error');
    });

    socket.on('status', (data) => {
        console.log('[Server]', data.message);
    });

    // ── Registration Events ──
    socket.on('registration_status', handleRegistrationStatus);

    // ── Authentication Events ──
    socket.on('auth_status', handleAuthStatus);
    socket.on('auth_complete', handleAuthComplete);

    // ── Processed Frame (from server) ──
    socket.on('processed_frame', handleProcessedFrame);
}

// ══════════════════════════════════════════════════════════════════════════════
//  WEBCAM MANAGEMENT
// ══════════════════════════════════════════════════════════════════════════════

async function startWebcam() {
    webcamVideo = document.getElementById('webcam-video');
    canvas = document.getElementById('webcam-canvas');
    processedFeed = document.getElementById('processed-feed');

    if (!webcamVideo || !canvas) {
        console.error('[Webcam] Video/canvas elements not found in DOM.');
        updateStatus('Camera elements not found.', 'error');
        return false;
    }

    ctx = canvas.getContext('2d');

    // Make sure video element is visible
    webcamVideo.style.display = 'block';
    webcamVideo.style.position = 'relative';
    webcamVideo.style.zIndex = '1';

    // Hide processed-feed initially (it overlays on top)
    if (processedFeed) {
        processedFeed.style.display = 'none';
    }

    // Hide the face guide during active capture
    const overlay = document.getElementById('camera-overlay');
    if (overlay) overlay.style.display = 'none';

    try {
        console.log('[Webcam] Requesting camera access...');
        updateCameraStatus('Requesting camera permission...');

        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });

        webcamVideo.srcObject = videoStream;

        // Wait for video to actually start playing
        await new Promise((resolve, reject) => {
            webcamVideo.onloadedmetadata = () => {
                webcamVideo.play().then(resolve).catch(reject);
            };
            setTimeout(() => reject(new Error('Video load timeout')), 5000);
        });

        // Set canvas size to match actual video
        canvas.width = webcamVideo.videoWidth || 640;
        canvas.height = webcamVideo.videoHeight || 480;

        console.log('[Webcam] Camera started. Resolution:', canvas.width, 'x', canvas.height);
        updateCameraStatus('Camera active');
        updateStatus('Camera started. Connecting to server...', 'info');
        return true;

    } catch (err) {
        console.error('[Webcam] Error:', err.name, err.message);
        let errorMsg = 'Camera error: ';
        if (err.name === 'NotAllowedError') {
            errorMsg += 'Permission denied. Please allow camera access and refresh.';
        } else if (err.name === 'NotFoundError') {
            errorMsg += 'No camera found. Please connect a webcam.';
        } else if (err.name === 'NotReadableError') {
            errorMsg += 'Camera is in use by another application.';
        } else {
            errorMsg += err.message;
        }
        updateStatus(errorMsg, 'error');
        updateCameraStatus('Camera unavailable');
        return false;
    }
}

function stopWebcam() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (webcamVideo) {
        webcamVideo.srcObject = null;
    }
}

function startStreaming(eventName) {
    if (isStreaming) return;
    isStreaming = true;

    console.log('[Stream] Starting frame stream:', eventName, 'at', FRAME_RATE, 'fps');

    let frameCount = 0;
    frameInterval = setInterval(() => {
        if (!isStreaming || !webcamVideo || webcamVideo.readyState < 2) {
            return;
        }
        if (!socket || !socket.connected) {
            console.warn('[Stream] Socket not connected, skipping frame.');
            return;
        }

        // Capture frame from video
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.6);
        const base64Data = frameData.replace(/^data:image\/jpeg;base64,/, '');

        // Send frame to server
        socket.emit(eventName, {
            frame: base64Data,
            sid: SESSION_ID
        });

        frameCount++;
        if (frameCount % 30 === 0) {
            console.log('[Stream] Sent', frameCount, 'frames');
        }

    }, 1000 / FRAME_RATE);
}

function stopStreaming() {
    isStreaming = false;
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    console.log('[Stream] Stopped.');
}

// ══════════════════════════════════════════════════════════════════════════════
//  REGISTRATION FLOW
// ══════════════════════════════════════════════════════════════════════════════

async function startRegistration(username) {
    console.log('[Registration] Starting for user:', username);

    // Step 1: Initialize socket
    initSocket();

    // Step 2: Start webcam
    updateStatus('Starting camera for face capture...', 'info');
    const cameraOk = await startWebcam();
    if (!cameraOk) {
        updateStatus('Camera failed. Please allow camera permissions and refresh.', 'error');
        return;
    }

    // Step 3: Wait for socket connection
    updateStatus('Camera ready. Connecting to server...', 'info');

    const waitForConnection = () => new Promise((resolve) => {
        if (socket && socket.connected) {
            console.log('[Registration] Socket already connected.');
            return resolve(true);
        }
        const onConnect = () => {
            socket.off('connect', onConnect);
            resolve(true);
        };
        socket.on('connect', onConnect);
        setTimeout(() => {
            socket.off('connect', onConnect);
            resolve(socket && socket.connected);
        }, 5000);
    });

    const connected = await waitForConnection();
    if (!connected) {
        updateStatus('Could not connect to server. Please refresh and try again.', 'error');
        return;
    }

    // Step 4: Tell server we're starting registration
    console.log('[Registration] Emitting start_registration');
    socket.emit('start_registration', {
        username: username,
        sid: SESSION_ID
    });

    // Step 5: Start streaming frames
    updateStatus('Look at the camera. Capturing face...', 'info');
    startStreaming('registration_frame');
}

function handleRegistrationStatus(data) {
    console.log('[Registration] Status:', data);

    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    if (data.status === 'capturing') {
        const pct = (data.captured / data.required) * 100;
        if (progressFill) progressFill.style.width = pct + '%';
        if (progressText) progressText.textContent = `${data.captured} / ${data.required} frames captured`;
        updateStatus(data.message, 'info');
    }
    else if (data.status === 'complete') {
        stopStreaming();
        stopWebcam();

        if (progressFill) progressFill.style.width = '100%';
        updateStatus('Registration successful! ' + data.message, 'success');

        // Complete registration via API
        fetch('/api/complete-registration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_encodings: data.num_encodings || 0 })
        })
        .then(res => res.json())
        .then(result => {
            if (result.success) {
                updateStatus('Registration complete! Redirecting to login...', 'success');
                setTimeout(() => {
                    window.location.href = '/login';
                }, 2000);
            } else {
                updateStatus('Error: ' + result.message, 'error');
            }
        })
        .catch(err => {
            console.error('[Registration] API error:', err);
            updateStatus('Registration failed. Please try again.', 'error');
        });
    }
    else if (data.status === 'error') {
        stopStreaming();
        updateStatus('Error: ' + data.message, 'error');
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  AUTHENTICATION FLOW
// ══════════════════════════════════════════════════════════════════════════════

async function startAuthentication(username) {
    console.log('[Auth] Starting for user:', username);

    initSocket();

    updateStatus('Starting camera for face authentication...', 'info');
    const cameraOk = await startWebcam();
    if (!cameraOk) {
        updateStatus('Camera failed. Please allow camera permissions and refresh.', 'error');
        return;
    }

    // Wait for socket connection
    updateStatus('Camera ready. Connecting to server...', 'info');

    const waitForConnection = () => new Promise((resolve) => {
        if (socket && socket.connected) return resolve(true);
        const onConnect = () => {
            socket.off('connect', onConnect);
            resolve(true);
        };
        socket.on('connect', onConnect);
        setTimeout(() => {
            socket.off('connect', onConnect);
            resolve(socket && socket.connected);
        }, 5000);
    });

    const connected = await waitForConnection();
    if (!connected) {
        updateStatus('Could not connect to server. Please refresh.', 'error');
        return;
    }

    // Tell server we're starting authentication
    console.log('[Auth] Emitting start_auth');
    socket.emit('start_auth', {
        username: username,
        sid: SESSION_ID
    });

    // Start streaming frames
    updateStatus('Look at the camera and blink naturally.', 'info');
    startStreaming('auth_frame');
}

function handleAuthStatus(data) {
    // Update module indicators
    updateIndicator('ind-face', data.face_ok);
    updateIndicator('ind-recog', data.recog_ok);
    updateIndicator('ind-live', data.live_ok);
    updateIndicator('ind-deepfake', data.deepfake_ok);

    // Update alert banner
    const alertBanner = document.getElementById('alert-banner');
    const alertText = document.getElementById('alert-text');

    if (data.alert && alertBanner) {
        alertBanner.classList.remove('hidden');
        if (alertText) alertText.textContent = `WARNING: ${data.alert_type} - ${data.reasons.join(' ')}`;
    }

    // Update status message
    if (data.decision === 'PENDING') {
        updateStatus(`Verifying... (Frame ${data.frame_count}) | Blinks: ${data.blinks}`, 'info');
    } else if (data.decision === 'DENIED') {
        updateStatus(`Checking: ${data.reasons.join(' | ')}`, 'warning');
    }
}

function handleAuthComplete(data) {
    stopStreaming();
    stopWebcam();

    if (data.status === 'granted') {
        updateStatus('ACCESS GRANTED: ' + data.message, 'success');

        // All indicators green
        ['ind-face', 'ind-recog', 'ind-live', 'ind-deepfake'].forEach(id => {
            updateIndicator(id, true);
        });

        // Complete face auth via API
        fetch('/api/complete-face-auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        })
        .then(res => res.json())
        .then(result => {
            if (result.success) {
                setTimeout(() => {
                    window.location.href = result.redirect;
                }, 1500);
            }
        })
        .catch(err => {
            console.error('[Auth] API error:', err);
        });
    } else {
        updateStatus('ACCESS DENIED: ' + data.message, 'error');
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  UI HELPERS
// ══════════════════════════════════════════════════════════════════════════════

function handleProcessedFrame(data) {
    // Display the server-processed frame (with bounding boxes etc.) overlaid on the raw video
    if (processedFeed && data.frame) {
        processedFeed.src = 'data:image/jpeg;base64,' + data.frame;
        processedFeed.style.display = 'block';
        processedFeed.style.position = 'absolute';
        processedFeed.style.top = '0';
        processedFeed.style.left = '0';
        processedFeed.style.zIndex = '2';
    }
}

function updateStatus(message, type) {
    const statusEl = document.getElementById('status-message');
    if (!statusEl) return;

    const iconEl = statusEl.querySelector('.status-icon');
    const textEl = statusEl.querySelector('.status-text');

    const icons = {
        'info': '⏳',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️'
    };

    if (iconEl) iconEl.textContent = icons[type] || '⏳';
    if (textEl) textEl.textContent = message;

    statusEl.className = 'status-message ' + (type || '');
}

function updateCameraStatus(message) {
    const el = document.getElementById('camera-status');
    if (el) el.textContent = message;
}

function updateIndicator(id, passed) {
    const el = document.getElementById(id);
    if (!el) return;

    el.classList.remove('passed', 'failed', 'checking');

    if (passed === true) {
        el.classList.add('passed');
    } else if (passed === false) {
        el.classList.add('checking');
    } else {
        el.classList.add('checking');
    }
}

// ── Auto-dismiss flash messages ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const flashes = document.querySelectorAll('.flash-message');
    flashes.forEach(flash => {
        setTimeout(() => {
            flash.style.opacity = '0';
            flash.style.transform = 'translateX(100%)';
            setTimeout(() => flash.remove(), 300);
        }, 5000);
    });
});
