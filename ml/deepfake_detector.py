"""
=============================================================================
 Module 4: Deepfake Detection — MesoNet + Texture/Frequency Analysis
 Multi-signal deepfake detection combining:
   1. MesoNet (Meso4 CNN) — lightweight neural network for face forgery
   2. Texture analysis — Local Binary Pattern + variance analysis
   3. Frequency domain — DCT spectral analysis for GAN artifacts

 Reference: MesoNet — a Compact Facial Video Forgery Detection Network
            (Afchar et al., 2018)
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

# ── Conditional import for TensorFlow ───────────────────────────────────────
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not available. Deepfake detection will use fallback.")


def build_meso4_model(input_shape=(256, 256, 3)):
    """
    Build the Meso4 architecture — a lightweight CNN for deepfake detection.

    Architecture:
    - 4 convolutional blocks with BatchNorm and MaxPool
    - Flatten → Dense(16) → Dropout → Dense(1, sigmoid)
    - Total params: ~28,000

    Args:
        input_shape (tuple): Input image dimensions (H, W, C).

    Returns:
        tf.keras.Model: Compiled Meso4 model.
    """
    if not TF_AVAILABLE:
        return None

    model = models.Sequential([
        # ── Block 1: 8 filters ──
        layers.Conv2D(8, (3, 3), padding='same', activation='relu',
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # ── Block 2: 8 filters ──
        layers.Conv2D(8, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # ── Block 3: 16 filters ──
        layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        # ── Block 4: 16 filters ──
        layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(4, 4), padding='same'),

        # ── Classifier ──
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # 0 = FAKE, 1 = REAL
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


class DeepfakeDetector:
    """
    Multi-signal deepfake detection engine.
    Combines three independent detection methods:
      1. MesoNet CNN (if TensorFlow + weights available)
      2. Texture analysis (LBP variance, Laplacian variance)
      3. Frequency domain analysis (DCT spectral features)

    The final confidence is a weighted fusion of all available signals.
    """

    def __init__(self):
        """Initialize the deepfake detector and load pretrained weights."""
        self.input_size = config.DEEPFAKE_INPUT_SIZE
        self.real_threshold = config.DEEPFAKE_REAL_THRESHOLD
        self.suspicious_threshold = config.DEEPFAKE_SUSPICIOUS_THRESHOLD
        self.model = None
        self.model_loaded = False
        self._lock = Lock()

        if TF_AVAILABLE:
            self.model = build_meso4_model(
                input_shape=(self.input_size[0], self.input_size[1], 3)
            )

            # ── Load pretrained weights if available ──
            weights_path = config.DEEPFAKE_MODEL_PATH
            if os.path.exists(weights_path):
                try:
                    self.model.load_weights(weights_path)
                    self.model_loaded = True
                    print("[INFO] Deepfake Detection: MesoNet weights loaded.")
                except Exception as e:
                    print(f"[WARNING] Could not load MesoNet weights: {e}")
                    print("[INFO] Using texture+frequency analysis (no CNN).")
            else:
                print(f"[INFO] MesoNet weights not found at {weights_path}")
                print("[INFO] Using texture+frequency analysis. Run:")
                print("       python scripts/train_demo_mesonet.py")
        else:
            print("[WARNING] TensorFlow not available. Using texture+frequency analysis.")

    def preprocess_face(self, face_image):
        """
        Preprocess a face crop for MesoNet input.

        Args:
            face_image (np.ndarray): Cropped face image (BGR).

        Returns:
            np.ndarray: Preprocessed image ready for model input.
        """
        # Resize to expected input size
        resized = cv2.resize(face_image, self.input_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)

        return batch

    # ═══════════════════════════════════════════════════════════════════════
    #  SIGNAL 1: MesoNet CNN
    # ═══════════════════════════════════════════════════════════════════════

    def _mesonet_score(self, face_image):
        """
        Get MesoNet CNN prediction.

        Returns:
            float or None: Confidence that face is REAL (0-1), or None if unavailable.
        """
        if self.model is None or not self.model_loaded:
            return None

        preprocessed = self.preprocess_face(face_image)

        with self._lock:
            prediction = self.model.predict(preprocessed, verbose=0)[0][0]

        return float(prediction)

    # ═══════════════════════════════════════════════════════════════════════
    #  SIGNAL 2: Texture Analysis
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _texture_score(face_image):
        """
        Analyze face texture for deepfake artifacts using:
        1. Laplacian variance — fake faces often have inconsistent blur
        2. Local Binary Pattern variance — texture consistency check
        3. Edge density analysis — real faces have natural edge distributions

        Returns:
            float: Score 0-1 where higher = more likely REAL.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # ── 1. Laplacian Variance (focus/blur consistency) ──
        # Real faces have consistent but moderate Laplacian variance.
        # Fake faces often have regions of unnatural sharpness or blur.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()

        # Normalize: typical real face lap_var is 100-800
        # Very low (< 50) = overly smooth (suspicious)
        # Very high (> 1500) = overly sharpened (suspicious)
        if lap_var < 30:
            lap_score = 0.3  # Too smooth — likely manipulated
        elif lap_var < 50:
            lap_score = 0.5
        elif lap_var > 1500:
            lap_score = 0.4  # Oversharpened
        elif lap_var > 800:
            lap_score = 0.6
        else:
            lap_score = 0.8 + 0.2 * min(1.0, lap_var / 500)
            lap_score = min(1.0, lap_score)

        # ── 2. Local texture consistency (LBP-like analysis) ──
        # Compute local variance in 8x8 blocks
        blocks = []
        block_size = 16
        for i in range(0, 128 - block_size, block_size):
            for j in range(0, 128 - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size].astype(np.float64)
                blocks.append(block.var())

        if blocks:
            block_vars = np.array(blocks)
            # Coefficient of variation of block variances
            # Real faces should have somewhat consistent texture variance
            cv_var = block_vars.std() / (block_vars.mean() + 1e-6)

            # High CV means inconsistent texture (suspicious)
            if cv_var > 2.0:
                texture_score = 0.4
            elif cv_var > 1.5:
                texture_score = 0.6
            elif cv_var > 1.0:
                texture_score = 0.75
            else:
                texture_score = 0.9
        else:
            texture_score = 0.5

        # ── 3. Edge density ──
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (128 * 128 * 255)

        # Natural edge density for faces: 0.02-0.15
        if edge_density < 0.01 or edge_density > 0.25:
            edge_score = 0.4  # Abnormal edge distribution
        elif edge_density < 0.02 or edge_density > 0.20:
            edge_score = 0.6
        else:
            edge_score = 0.85

        # Weighted combination
        final_score = 0.4 * lap_score + 0.35 * texture_score + 0.25 * edge_score
        return float(np.clip(final_score, 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════════
    #  SIGNAL 3: Frequency Domain Analysis (DCT)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _frequency_score(face_image):
        """
        Analyze face in frequency domain using DCT (Discrete Cosine Transform).
        GAN-generated faces often have characteristic spectral artifacts:
        - Unusual energy concentration in specific frequency bands
        - Missing high-frequency natural noise patterns
        - Periodic spectral peaks from upsampling

        Returns:
            float: Score 0-1 where higher = more likely REAL.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128)).astype(np.float64)

        # Compute 2D DCT
        dct = cv2.dct(gray)
        dct_log = np.log1p(np.abs(dct))

        # ── Analyze frequency band energy distribution ──
        h, w = dct_log.shape

        # Low frequency (top-left quadrant)
        low_freq = dct_log[:h // 4, :w // 4].mean()

        # Mid frequency
        mid_freq = dct_log[h // 4:h // 2, w // 4:w // 2].mean()

        # High frequency (bottom-right)
        high_freq = dct_log[h // 2:, w // 2:].mean()

        # ── Ratio analysis ──
        # Real faces: gradual energy dropoff from low to high
        # Fake faces: abnormal energy in certain bands, or no high-freq content

        if low_freq > 0:
            mid_ratio = mid_freq / low_freq
            high_ratio = high_freq / low_freq
        else:
            return 0.5  # Can't analyze

        # Natural mid-to-low ratio: 0.3-0.7
        if 0.25 < mid_ratio < 0.8:
            mid_score = 0.85
        elif 0.15 < mid_ratio < 0.9:
            mid_score = 0.65
        else:
            mid_score = 0.4  # Abnormal frequency distribution

        # Natural high-to-low ratio: 0.1-0.4
        if 0.05 < high_ratio < 0.5:
            high_score = 0.85
        elif high_ratio < 0.03:
            high_score = 0.3  # Suspiciously missing high-freq (oversmoothed)
        elif high_ratio > 0.6:
            high_score = 0.4  # Too much high-freq noise
        else:
            high_score = 0.6

        # ── Check for periodic patterns (GAN artifact) ──
        # Look for unusual peaks in the DCT spectrum
        dct_flat = dct_log.flatten()
        dct_sorted = np.sort(dct_flat)[::-1]

        # Ratio of top peaks to median — very high ratio = periodic artifact
        top_peaks = dct_sorted[:10].mean()
        median_val = np.median(dct_flat)
        peak_ratio = top_peaks / (median_val + 1e-6)

        if peak_ratio > 15:
            periodic_score = 0.4  # Strong periodic artifacts
        elif peak_ratio > 10:
            periodic_score = 0.6
        else:
            periodic_score = 0.85

        final_score = 0.35 * mid_score + 0.35 * high_score + 0.30 * periodic_score
        return float(np.clip(final_score, 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════════
    #  MAIN ANALYSIS FUNCTION
    # ═══════════════════════════════════════════════════════════════════════

    def analyze_face(self, face_image):
        """
        Analyze a face image for deepfake manipulation using multi-signal fusion.

        Signal weights:
        - MesoNet CNN:        50% (if available)
        - Texture analysis:   30% (always available)
        - Frequency analysis: 20% (always available)

        If MesoNet is not available, texture and frequency scores are
        re-weighted to 60% and 40% respectively.

        Args:
            face_image (np.ndarray): Cropped face image (BGR).

        Returns:
            dict: Analysis result containing:
                - 'is_real': bool — whether classified as real
                - 'is_suspicious': bool — whether the confidence is borderline
                - 'confidence_real': float — confidence that face is REAL (0-1)
                - 'confidence_fake': float — confidence that face is FAKE (0-1)
                - 'label': str — 'REAL', 'FAKE', or 'SUSPICIOUS'
                - 'signals': dict — individual signal scores
        """
        if face_image is None or face_image.size == 0:
            return {
                'is_real': False, 'is_suspicious': True,
                'confidence_real': 0.0, 'confidence_fake': 1.0,
                'label': 'ERROR', 'signals': {}
            }

        signals = {}

        # ── Signal 1: MesoNet ──
        mesonet_score = self._mesonet_score(face_image)
        if mesonet_score is not None:
            signals['mesonet'] = mesonet_score

        # ── Signal 2: Texture ──
        texture_score = self._texture_score(face_image)
        signals['texture'] = texture_score

        # ── Signal 3: Frequency ──
        frequency_score = self._frequency_score(face_image)
        signals['frequency'] = frequency_score

        # ── Weighted Fusion ──
        if mesonet_score is not None:
            # All three signals available
            confidence_real = (
                0.50 * mesonet_score +
                0.30 * texture_score +
                0.20 * frequency_score
            )
        else:
            # Fallback: texture + frequency only
            confidence_real = (
                0.60 * texture_score +
                0.40 * frequency_score
            )

        confidence_fake = 1.0 - confidence_real

        # ── Classification ──
        if confidence_real >= self.real_threshold:
            label = "REAL"
            is_real = True
            is_suspicious = False
        elif confidence_real >= self.suspicious_threshold:
            label = "SUSPICIOUS"
            is_real = False
            is_suspicious = True
        else:
            label = "FAKE"
            is_real = False
            is_suspicious = False

        return {
            'is_real': is_real,
            'is_suspicious': is_suspicious,
            'confidence_real': float(confidence_real),
            'confidence_fake': float(confidence_fake),
            'label': label,
            'signals': signals
        }

    def get_model_summary(self):
        """Return model architecture summary as string."""
        if self.model is not None:
            string_list = []
            self.model.summary(print_fn=lambda x: string_list.append(x))
            return "\n".join(string_list)
        return "Model not available."

    # ═══════════════════════════════════════════════════════════════════════
    #  GRAD-CAM EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════════════════

    def generate_gradcam(self, face_image, layer_name=None):
        """
        Generate a Grad-CAM heatmap showing which regions of the face
        activated the deepfake/real classification.

        Grad-CAM (Gradient-weighted Class Activation Mapping) uses the
        gradients of the prediction flowing into the final convolutional
        layer to produce a localization map highlighting important regions.

        Args:
            face_image (np.ndarray): Cropped face image (BGR).
            layer_name (str): Name of the conv layer to visualize.
                              Defaults to the last conv layer.

        Returns:
            dict: {
                'heatmap': np.ndarray — colored heatmap (BGR, same size as input),
                'overlay': np.ndarray — original image with heatmap overlay,
                'heatmap_b64': str — base64-encoded JPEG of overlay,
                'prediction': float — model prediction (0=fake, 1=real),
                'label': str — 'REAL' or 'FAKE'
            }
            or None if model is not available.
        """
        if not TF_AVAILABLE or self.model is None or not self.model_loaded:
            return None

        import tensorflow as tf

        # Preprocess
        preprocessed = self.preprocess_face(face_image)

        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break

        if layer_name is None:
            return None

        # Create gradient model
        try:
            grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[
                    self.model.get_layer(layer_name).output,
                    self.model.output
                ]
            )
        except ValueError:
            return None

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed)
            pred_value = predictions[0][0]

        # Gradients of the prediction w.r.t. the conv layer output
        grads = tape.gradient(pred_value, conv_outputs)

        if grads is None:
            return None

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature maps by the gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to match input image
        h, w = face_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        # Create overlay
        overlay = cv2.addWeighted(face_image, 0.6, heatmap_colored, 0.4, 0)

        # Convert overlay to base64
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        import base64
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        prediction = float(pred_value.numpy())
        label = "REAL" if prediction >= self.real_threshold else "FAKE"

        return {
            'heatmap': heatmap_colored,
            'overlay': overlay,
            'heatmap_b64': heatmap_b64,
            'prediction': prediction,
            'label': label
        }

    def predict_single_image(self, image_path_or_array):
        """
        Predict whether a single image is real or fake.
        Convenience method for the web upload endpoint.

        Args:
            image_path_or_array: File path (str) or BGR numpy array.

        Returns:
            dict: Full analysis result including Grad-CAM if available.
        """
        if isinstance(image_path_or_array, str):
            face_image = cv2.imread(image_path_or_array)
            if face_image is None:
                return {'error': 'Could not read image file.'}
        else:
            face_image = image_path_or_array

        # Get main analysis
        result = self.analyze_face(face_image)

        # Add Grad-CAM if model is available
        gradcam = self.generate_gradcam(face_image)
        if gradcam:
            result['gradcam_b64'] = gradcam['heatmap_b64']
            result['gradcam_label'] = gradcam['label']
            result['gradcam_prediction'] = gradcam['prediction']

        return result


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = DeepfakeDetector()
    print(detector.get_model_summary())

    # Test with a random image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = detector.analyze_face(test_img)
    print(f"\nTest result: {result}")

    # Test Grad-CAM
    gradcam = detector.generate_gradcam(test_img)
    if gradcam:
        print(f"Grad-CAM generated: label={gradcam['label']}, pred={gradcam['prediction']:.4f}")
    else:
        print("Grad-CAM not available (model not loaded).")
