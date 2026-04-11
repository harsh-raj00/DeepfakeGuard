"""
=============================================================================
 Module 1: Face Detection — OpenCV DNN (SSD + ResNet-10)
 Provides real-time face detection from video frames using a pre-trained
 Single Shot Detector (SSD) with a ResNet-10 backbone.
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
from threading import Lock

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FaceDetector:
    """
    Real-time face detector using OpenCV's DNN module.
    Uses the SSD (Single Shot Detector) framework with a ResNet-10 backbone
    trained on face data. Runs at 30+ FPS on CPU.

    NOTE: OpenCV DNN net.forward() is NOT thread-safe. A Lock is used to
    serialize concurrent inference calls from Flask-SocketIO threads.
    """

    def __init__(self, confidence_threshold=None):
        """
        Initialize the face detector by loading the pre-trained model.

        Args:
            confidence_threshold (float): Minimum detection confidence (0-1).
                                          Defaults to config value.
        """
        self.confidence_threshold = confidence_threshold or config.FACE_DETECTION_CONFIDENCE
        self._lock = Lock()  # Thread-safety for net.forward()

        # ── Load the pre-trained SSD model ──
        proto_path = config.FACE_DETECTION_MODEL_PROTO
        weights_path = config.FACE_DETECTION_MODEL_WEIGHTS

        if not os.path.exists(proto_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Face detection model files not found.\n"
                f"Expected:\n  {proto_path}\n  {weights_path}\n"
                f"Run 'python scripts/download_models.py' to download them."
            )

        self.net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        print("[INFO] Face Detection model loaded successfully.")

    def detect_faces(self, frame):
        """
        Detect faces in a single frame.

        Args:
            frame (np.ndarray): BGR image from webcam.

        Returns:
            list of dict: Each dict contains:
                - 'box': (x, y, w, h) bounding box coordinates
                - 'confidence': detection confidence score
                - 'roi': cropped face region (BGR)
        """
        (h, w) = frame.shape[:2]
        detections = []

        # ── Preprocess: create a blob from the input frame ──
        # SSD expects 300x300 input with mean subtraction
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),  # Resize to 300x300
            1.0,                             # Scale factor
            (300, 300),                      # Spatial size
            (104.0, 177.0, 123.0),          # Mean subtraction (BGR)
            swapRB=False,                    # Already BGR
            crop=False
        )

        # ── Forward pass through the network (thread-safe) ──
        with self._lock:
            self.net.setInput(blob)
            raw_detections = self.net.forward()

        # ── Process each detection ──
        for i in range(raw_detections.shape[2]):
            confidence = raw_detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence < self.confidence_threshold:
                continue

            # Scale bounding box back to frame dimensions
            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp coordinates to frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Skip invalid boxes
            if startX >= endX or startY >= endY:
                continue

            # Extract the face ROI
            face_roi = frame[startY:endY, startX:endX]

            detections.append({
                'box': (startX, startY, endX - startX, endY - startY),
                'confidence': float(confidence),
                'roi': face_roi
            })

        return detections

    def draw_detections(self, frame, detections, label=None, color=(0, 255, 0)):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame (np.ndarray): Original frame to annotate.
            detections (list): Detection results from detect_faces().
            label (str): Optional label to display.
            color (tuple): BGR color for the bounding box.

        Returns:
            np.ndarray: Annotated frame.
        """
        annotated = frame.copy()

        for det in detections:
            (x, y, w, h) = det['box']
            conf = det['confidence']

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Prepare label text
            text = label if label else f"Face: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Draw label background
            cv2.rectangle(
                annotated,
                (x, y - text_size[1] - 10),
                (x + text_size[0], y),
                color, -1
            )

            # Draw label text
            cv2.putText(
                annotated, text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2
            )

        return annotated


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = FaceDetector()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    print("[INFO] Starting face detection test. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        frame = detector.draw_detections(frame, faces)

        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
