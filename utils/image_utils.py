"""
=============================================================================
 Image Utilities — Preprocessing helpers for the ML pipeline.
=============================================================================
"""

import cv2
import numpy as np
import base64


def resize_frame(frame, width=640, height=480):
    """Resize a frame to the specified dimensions."""
    return cv2.resize(frame, (width, height))


def frame_to_base64(frame, quality=80):
    """
    Convert an OpenCV frame (BGR) to a base64-encoded JPEG string.
    Used for sending frames over WebSocket to the browser.

    Args:
        frame (np.ndarray): BGR frame from OpenCV.
        quality (int): JPEG compression quality (1-100).

    Returns:
        str: Base64-encoded JPEG string.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_frame(b64_string):
    """
    Convert a base64-encoded image string back to an OpenCV frame.

    Args:
        b64_string (str): Base64-encoded image data.

    Returns:
        np.ndarray: BGR frame.
    """
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def preprocess_face_crop(face_roi, target_size=(256, 256)):
    """
    Preprocess a cropped face region for model input.

    Args:
        face_roi (np.ndarray): Cropped face image (BGR).
        target_size (tuple): Desired output size (W, H).

    Returns:
        np.ndarray: Preprocessed face image.
    """
    if face_roi is None or face_roi.size == 0:
        return None

    # Resize to target dimensions
    resized = cv2.resize(face_roi, target_size)

    return resized


def add_padding_to_box(box, frame_shape, padding=20):
    """
    Add padding to a bounding box while keeping it within frame boundaries.

    Args:
        box (tuple): (x, y, w, h) bounding box.
        frame_shape (tuple): (height, width) of the frame.
        padding (int): Pixels of padding to add.

    Returns:
        tuple: Padded (x, y, w, h) bounding box.
    """
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_w - x, w + 2 * padding)
    h = min(frame_h - y, h + 2 * padding)

    return (x, y, w, h)


def draw_status_bar(frame, status_text, color=(0, 255, 0)):
    """
    Draw a semi-transparent status bar at the top of the frame.

    Args:
        frame (np.ndarray): Frame to annotate.
        status_text (str): Text to display.
        color (tuple): BGR color for the bar.

    Returns:
        np.ndarray: Annotated frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Text
    cv2.putText(annotated, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated
