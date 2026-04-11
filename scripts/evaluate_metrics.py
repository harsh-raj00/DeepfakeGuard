"""
=============================================================================
 Evaluation Metrics Calculator
 Computes Accuracy, Precision, Recall, F1-Score, FAR (False Acceptance Rate),
 and FRR (False Rejection Rate) for the authentication system.

 Also generates a visual confusion matrix and detailed test report.
=============================================================================
"""

import os
import sys
import json
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.db_utils import get_login_history, get_audit_logs


def calculate_metrics(login_records=None):
    """
    Calculate authentication evaluation metrics from login history.

    If no records are provided, loads from the database.

    Expected record format:
        {'status': 'SUCCESS'|'DENIED'|'ALERT', 'username': '...', ...}

    In a real evaluation, we need ground truth. Here we use:
    - 'SUCCESS' by a real user = True Positive (TP)
    - 'DENIED' for an impostor/fake = True Negative (TN)
    - 'SUCCESS' for an impostor/fake = False Positive (FP) — False Acceptance
    - 'DENIED' for a real user = False Negative (FN) — False Rejection

    For this demo, we simulate by treating:
    - SUCCESS without alert = TP (authentic user accepted)
    - DENIED with alert = TN (impostor/fake rejected correctly)
    - SUCCESS with alert = FP (should not have accepted)
    - DENIED without alert = FN (legitimate user rejected)

    Returns:
        dict: All computed metrics.
    """
    if login_records is None:
        login_records = get_login_history(limit=1000)

    if not login_records:
        print("[INFO] No login records found. Using sample data for demonstration.")
        # Generate sample data for demonstration
        login_records = generate_sample_data()

    # ── Classify each record ──
    TP = 0  # True Positive: Real user authenticated correctly
    TN = 0  # True Negative: Impostor/fake rejected correctly
    FP = 0  # False Positive: Impostor/fake incorrectly accepted (FAR)
    FN = 0  # False Negative: Real user incorrectly rejected (FRR)

    for record in login_records:
        status = record.get('status', '')
        has_alert = record.get('alert_type') is not None and record.get('alert_type') != ''

        if status == 'SUCCESS' and not has_alert:
            TP += 1
        elif status == 'DENIED' and has_alert:
            TN += 1
        elif status == 'SUCCESS' and has_alert:
            FP += 1  # False Acceptance
        elif status == 'DENIED' and not has_alert:
            FN += 1

    # ── Calculate Metrics ──
    total = TP + TN + FP + FN

    # Accuracy = (TP + TN) / Total
    accuracy = (TP + TN) / total if total > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall (Sensitivity) = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # FAR (False Acceptance Rate) = FP / (FP + TN)
    far = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # FRR (False Rejection Rate) = FN / (FN + TP)
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    metrics = {
        'confusion_matrix': {
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        },
        'total_records': total,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'far': round(far, 4),
        'frr': round(frr, 4),
        'specificity': round(specificity, 4),
        'timestamp': datetime.now().isoformat()
    }

    return metrics


def generate_sample_data():
    """
    Generate realistic sample login records for evaluation demonstration.
    Simulates a comprehensive mix across all test scenarios:

    - 80 genuine user logins (95% success rate)
    - 30 photo attack attempts (97% rejection rate)
    - 20 deepfake attacks (95% rejection rate)
    - 20 unknown persons (100% rejection rate)

    Total: 150 records representing realistic authentication scenarios.
    """
    import random
    random.seed(42)  # Reproducible results

    records = []

    # ── Scenario 1: Genuine user logins (mostly success) ──
    # Real registered users attempting legitimate authentication
    for i in range(80):
        # 95% success — some fail due to poor lighting, angle, etc.
        is_success = random.random() > 0.05
        records.append({
            'username': f'user_{random.randint(1, 5)}',
            'status': 'SUCCESS' if is_success else 'DENIED',
            'alert_type': None,  # No alert — legitimate attempt
            'face_confidence': round(random.uniform(0.70, 0.95), 3),
            'liveness_blinks': random.randint(1, 4),
            'deepfake_confidence': round(random.uniform(0.80, 0.98), 3),
            'details': 'All checks passed.' if is_success else 'Liveness check timeout.'
        })

    # ── Scenario 2: Photo attack attempts ──
    # Attacker presents a printed/screen photo of a registered user
    for i in range(30):
        # 97% correctly rejected — rare false accepts
        is_rejected = random.random() > 0.03
        records.append({
            'username': f'user_{random.randint(1, 5)}',
            'status': 'DENIED' if is_rejected else 'SUCCESS',
            'alert_type': 'POSSIBLE_PHOTO_ATTACK',
            'face_confidence': round(random.uniform(0.50, 0.80), 3),
            'liveness_blinks': 0,  # Photos don't blink
            'deepfake_confidence': round(random.uniform(0.30, 0.60), 3),
            'details': 'No blink detected - possible photo attack.' if is_rejected else 'False acceptance'
        })

    # ── Scenario 3: Deepfake attacks ──
    # GAN-generated or face-swapped video presented
    for i in range(20):
        # 95% correctly rejected
        is_rejected = random.random() > 0.05
        records.append({
            'username': f'attacker_{random.randint(1, 3)}',
            'status': 'DENIED' if is_rejected else 'SUCCESS',
            'alert_type': 'DEEPFAKE_DETECTED',
            'face_confidence': round(random.uniform(0.60, 0.90), 3),
            'liveness_blinks': random.randint(0, 2),
            'deepfake_confidence': round(random.uniform(0.10, 0.45), 3),
            'details': 'Deepfake manipulation detected.' if is_rejected else 'Sophisticated deepfake bypassed detection'
        })

    # ── Scenario 4: Unknown person attempts ──
    # Unregistered person tries to authenticate as a known user
    for i in range(20):
        # 100% correctly rejected — face doesn't match
        records.append({
            'username': 'unknown',
            'status': 'DENIED',
            'alert_type': None,
            'face_confidence': round(random.uniform(0.10, 0.40), 3),
            'liveness_blinks': random.randint(0, 3),
            'deepfake_confidence': round(random.uniform(0.70, 0.95), 3),
            'details': 'Face does not match any registered user.'
        })

    random.shuffle(records)
    return records


def print_metrics_report(metrics):
    """Print a comprehensive formatted metrics report."""
    print("\n" + "=" * 65)
    print("  FACE AUTHENTICATION SYSTEM — EVALUATION REPORT")
    print("=" * 65)
    print(f"  Generated:     {metrics['timestamp']}")
    print(f"  Total Records: {metrics['total_records']}")
    print("-" * 65)

    # ── Confusion Matrix ──
    print("\n  CONFUSION MATRIX")
    print("  " + "-" * 45)
    cm = metrics['confusion_matrix']
    print(f"                        Predicted")
    print(f"                     Accept  |  Reject")
    print(f"   Actual Accept  | TP={cm['TP']:>4}  | FN={cm['FN']:>4}")
    print(f"   Actual Reject  | FP={cm['FP']:>4}  | TN={cm['TN']:>4}")
    print("  " + "-" * 45)

    # ── Performance Metrics ──
    print("\n  PERFORMANCE METRICS")
    print("  " + "-" * 45)
    print(f"   Accuracy:    {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:   {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:      {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:    {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")

    # ── Security Metrics ──
    print("\n  SECURITY METRICS")
    print("  " + "-" * 45)
    print(f"   FAR (False Acceptance Rate): {metrics['far']:.4f}  ({metrics['far']*100:.2f}%)")
    print(f"   FRR (False Rejection Rate):  {metrics['frr']:.4f}  ({metrics['frr']*100:.2f}%)")
    print(f"   Specificity:                 {metrics['specificity']:.4f}  ({metrics['specificity']*100:.2f}%)")

    # ── Visual bar charts ──
    print("\n  VISUAL SUMMARY")
    print("  " + "-" * 45)
    max_bar = 40
    metrics_bars = [
        ("Accuracy ", metrics['accuracy']),
        ("Precision", metrics['precision']),
        ("Recall   ", metrics['recall']),
        ("F1-Score ", metrics['f1_score']),
        ("Specific.", metrics['specificity']),
    ]
    for name, val in metrics_bars:
        bar_len = int(val * max_bar)
        bar = "#" * bar_len + "." * (max_bar - bar_len)
        print(f"   {name} |{bar}| {val*100:.1f}%")

    print("\n" + "=" * 65)

    # ── Test Scenario Breakdown ──
    print("\n  TEST SCENARIO BREAKDOWN")
    print("  " + "-" * 45)
    print(f"   Genuine Users:    ~80 records (TP + FN = {cm['TP'] + cm['FN']})")
    print(f"   Photo Attacks:    ~30 records")
    print(f"   Deepfake Attacks: ~20 records")
    print(f"   Unknown Persons:  ~20 records")
    print(f"   Total:            {metrics['total_records']} records")

    print("\n" + "=" * 65)

    # Save report to file
    report_path = os.path.join(config.DATA_DIR, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    metrics = calculate_metrics()
    print_metrics_report(metrics)
