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
    Real-time face detector using OpenCV's Haar Cascade classifier.
    Replaces the original SSD DNN approach to ensure true positives are
    not rejected often, providing a robust initial matching step before deepfake classification.
    """

    def __init__(self, confidence_threshold=None):
        """
        Initialize the face detector by loading the pre-trained Haar Cascade model.
        """
        self.confidence_threshold = confidence_threshold or config.FACE_DETECTION_CONFIDENCE
        self._lock = Lock()  # Thread-safety 

        # ── Load the Haar Cascade model ──
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.cascade.empty():
            raise FileNotFoundError(f"Haar cascade model not found at {cascade_path}")
            
        print("[INFO] Haar Cascade Face Detection model loaded successfully.")

    def detect_faces(self, frame):
        """
        Detect faces in a single frame using Haar cascade.

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
        
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect faces (thread-safe) ──
        with self._lock:
            # outputRejectLevels allows us to extract pseudo-confidence weights
            faces, rejectLevels, levelWeights = self.cascade.detectMultiScale3(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=4, 
                minSize=(30, 30),
                outputRejectLevels=True
            )

        if len(faces) == 0:
            return detections

        # ── Process each detection ──
        for i, (startX, startY, boxW, boxH) in enumerate(faces):
            if levelWeights is not None:
                # Depending on OpenCV version, levelWeights is either a 1D array or 2D array of shape (N, 1)
                try:
                    weight = float(levelWeights[i][0])
                except (IndexError, TypeError):
                    weight = float(levelWeights[i])
            else:
                weight = 1.0
            
            # Map cascade weight to a pseudo-confidence [0, 1]
            confidence = min(1.0, max(0.0, weight / 5.0))

            if confidence < (self.confidence_threshold - 0.2):  # Slightly relax threshold for Haar
                continue

            # Clamp coordinates to frame boundaries
            endX = startX + boxW
            endY = startY + boxH
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
                'confidence': confidence,
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
