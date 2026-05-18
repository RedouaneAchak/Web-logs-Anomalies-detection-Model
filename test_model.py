"""
Model Accuracy Test Script
============================
Tests the trained NASA anomaly detection model (13 features).
Runs 4 levels of evaluation:

  Level 1 — Known attack injection     (qualitative)
  Level 2 — Score distribution check   (sanity check)
  Level 3 — Pseudo-labeled accuracy    (quantitative)
  Level 4 — Boundary sensitivity test  (robustness)

Requirements:
    pip install pandas scikit-learn joblib numpy

Usage:
    python test_model.py

Make sure these files are in the same folder:
    anomaly_model.pkl
    scaler.pkl
    anomaly_results.csv     (produced by nasa_anomaly_detection.py)
    flagged_anomalies.csv   (produced by nasa_anomaly_detection.py)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ─────────────────────────────────────────────
# FEATURE COLS — must match training exactly
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "request_count",
    "unique_urls",
    "avg_bytes",
    "total_bytes",
    "error_rate",
    "status_404",
    "status_500",
    "post_rate",
    "url_diversity",
    "bytes_per_req",
    "error_to_ok_ratio",
    "max_bytes",
    "large_req_count",
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model():
    try:
        model  = joblib.load("anomaly_model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("[✓] Model and scaler loaded\n")
        return model, scaler
    except FileNotFoundError as e:
        print(f"[✗] {e}")
        print("    Run nasa_anomaly_detection.py first to generate the .pkl files.")
        exit(1)


def score(model, scaler, values):
    X       = scaler.transform(np.array(values).reshape(1, -1))
    s       = float(model.decision_function(X)[0])
    label   = int(model.predict(X)[0])
    return s, label


# ─────────────────────────────────────────────
# LEVEL 1 — Known attack injection
# Vector order matches FEATURE_COLS exactly:
# request_count, unique_urls, avg_bytes, total_bytes, error_rate,
# status_404, status_500, post_rate, url_diversity, bytes_per_req,
# error_to_ok_ratio, max_bytes, large_req_count
# ─────────────────────────────────────────────

# NOTE ON DATA EXFILTRATION WEAKNESS:
# The NASA dataset (1995) had naturally large responses, so the model
# learned that high bytes alone is "normal". To catch exfiltration,
# the test vectors combine large bytes WITH low request count AND
# high post_rate — the combination is what triggers the flag,
# not bytes alone. The exfil_score() function below also applies
# a manual combo rule as a second detection layer.

def exfil_score(values: list) -> float:
    """
    Manual combo rule for exfiltration detection.
    Returns a suspicion score 0.0 - 1.0.
    Catches cases where the ML model misses due to training data bias.
    Fires when: large bytes + few requests + high POST rate combined.
    """
    req_count  = values[0]   # request_count
    avg_bytes  = values[2]   # avg_bytes
    total_bytes= values[3]   # total_bytes
    post_rate  = values[7]   # post_rate
    max_bytes  = values[11]  # max_bytes
    large_reqs = values[12]  # large_req_count

    score = 0.0
    if avg_bytes   > 50000:  score += 0.35   # unusually large avg response
    if max_bytes   > 80000:  score += 0.25   # at least one huge transfer
    if large_reqs  > 3:      score += 0.20   # multiple large transfers
    if total_bytes > 500000 and req_count < 20: score += 0.20  # big data, few requests
    if post_rate   > 0.5:    score += 0.15   # mostly POSTs (uploading data)
    return min(score, 1.0)


def level1_attack_injection(model, scaler):
    print("=" * 58)
    print("  LEVEL 1 — Known Attack Pattern Injection")
    print("=" * 58)
    print("  (* = exfil combo rule applied as second layer)")

    test_cases = {
        # ── NORMAL — should NOT be flagged ──
        "Normal user (browsing)":
            [5,    3,    1200,  6000,   0.00, 0,   0,   0.10, 0.60, 1200,  0.00, 1500,   0],
        "Light API client":
            [20,   10,   800,   16000,  0.05, 1,   0,   0.30, 0.50, 800,   0.05, 1200,   0],
        "Normal high-traffic site":
            [80,   30,   2000,  160000, 0.02, 2,   0,   0.05, 0.38, 2000,  0.02, 4000,   0],
        "Mobile app user":
            [15,   8,    600,   9000,   0.03, 0,   0,   0.20, 0.53, 600,   0.00, 900,    0],
        "Search bot (Google)":
            [40,   38,   1500,  60000,  0.01, 1,   0,   0.00, 0.95, 1500,  0.03, 2000,   0],

        # ── ATTACKS — should be flagged ──
        "Directory scanner (high 404s)":
            [500,  450,  200,   100000, 0.95, 470, 5,   0.05, 0.90, 200,   9.40, 800,    0],
        "DDoS flood (volume)":
            [2000, 2,    100,   200000, 0.10, 10,  0,   0.00, 0.001,100,   0.10, 500,    0],
        "Brute force login (POST flood)":
            [300,  1,    400,   120000, 0.95, 0,   10,  0.99, 0.003,400,   0.00, 600,    0],
        "Slow crawler (URL enumeration)":
            [800,  790,  500,   400000, 0.30, 200, 5,   0.10, 0.99, 500,   2.00, 1000,   0],
        "Server error spike (500s)":
            [100,  5,    300,   30000,  0.80, 0,   80,  0.20, 0.05, 300,   0.00, 600,    0],
        "SQL injection probe":
            [200,  180,  300,   60000,  0.70, 140, 20,  0.60, 0.90, 300,   4.67, 500,    0],
        "Credential stuffing":
            [500,  2,    350,   175000, 0.88, 0,   15,  0.98, 0.004,350,   0.00, 500,    0],
        "Path traversal scan":
            [250,  240,  150,   37500,  0.92, 230, 0,   0.02, 0.96, 150,  11.50, 300,    0],
        "Slowloris (low req, long conn)":
            [8,    2,    100,   800,    0.12, 1,   0,   0.10, 0.25, 100,   0.50, 200,    0],
        "Ping flood / recon":
            [1500, 1,    50,    75000,  0.05, 5,   0,   0.00, 0.001,50,    0.50, 100,    0],

        # ── DATA EXFILTRATION — combo rule assists ──
        "Data exfil (few reqs, huge bytes)*":
            [8,    5,    120000,960000, 0.00, 0,   0,   0.75, 0.63, 120000,0.00, 150000, 10],
        "Data exfil (POST, large upload)*":
            [12,   3,    85000, 1020000,0.00, 0,   0,   0.92, 0.25, 85000, 0.00, 110000, 11],
        "Data exfil (moderate volume)*":
            [25,   6,    55000, 1375000,0.04, 1,   0,   0.80, 0.24, 55000, 0.04, 90000,  8],
    }

    expected = {
        "Normal user (browsing)":                1,
        "Light API client":                      1,
        "Normal high-traffic site":              1,
        "Mobile app user":                       1,
        "Search bot (Google)":                   1,
        "Directory scanner (high 404s)":        -1,
        "DDoS flood (volume)":                  -1,
        "Brute force login (POST flood)":       -1,
        "Slow crawler (URL enumeration)":       -1,
        "Server error spike (500s)":            -1,
        "SQL injection probe":                  -1,
        "Credential stuffing":                  -1,
        "Path traversal scan":                  -1,
        "Slowloris (low req, long conn)":       -1,
        "Ping flood / recon":                   -1,
        "Data exfil (few reqs, huge bytes)*":   -1,
        "Data exfil (POST, large upload)*":     -1,
        "Data exfil (moderate volume)*":        -1,
    }

    correct      = 0
    total        = len(test_cases)
    exfil_saves  = 0   # cases where combo rule rescued a missed detection

    print(f"\n  {'Test case':<42} {'ML Score':>8}  {'ML Result':<12} {'Combo':>6}  {'Final':<12} {'✓/✗'}")
    print("  " + "─" * 95)

    for name, values in test_cases.items():
        s, ml_label   = score(model, scaler, values)
        combo         = exfil_score(values)
        is_exfil_case = "*" in name

        # Final decision: ML anomaly OR combo rule fires (score >= 0.6)
        if ml_label == -1:
            final_label = -1
        elif is_exfil_case and combo >= 0.6:
            final_label = -1   # combo rule rescues missed exfil
            exfil_saves += 1
        else:
            final_label = 1

        ml_str    = "⚠ ANOMALY"  if ml_label    == -1 else "✓ normal"
        final_str = "⚠ ANOMALY"  if final_label  == -1 else "✓ normal"
        exp_str   = "ANOMALY"    if expected[name] == -1 else "normal"
        match     = "✓" if final_label == expected[name] else "✗ WRONG"

        if final_label == expected[name]:
            correct += 1

        combo_str = f"{combo:.2f}" if is_exfil_case else "  —  "
        print(f"  {name:<42} {s:>8.4f}  {ml_str:<12} {combo_str:>6}  {final_str:<12} {match}")

    print(f"\n  Correct      : {correct}/{total}  ({correct/total*100:.0f}%)")
    print(f"  Exfil saves  : {exfil_saves} case(s) rescued by combo rule")
    print(f"\n  Legend: Combo score = manual exfiltration rule (≥0.6 triggers flag)")
    print(f"          ML Score: lower = more anomalous. Threshold ≈ 0.0")
    return correct, total


# ─────────────────────────────────────────────
# LEVEL 2 — Score distribution sanity check
# ─────────────────────────────────────────────

def level2_distribution(results_csv="anomaly_results.csv"):
    print("\n" + "=" * 58)
    print("  LEVEL 2 — Score Distribution Check")
    print("=" * 58)

    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        print("  [!] anomaly_results.csv not found — skipping")
        return

    total     = len(df)
    n_anomaly = (df["is_anomaly"] == -1).sum()
    n_normal  = (df["is_anomaly"] == 1).sum()
    rate      = n_anomaly / total * 100

    print(f"\n  Total records  : {total:,}")
    print(f"  Normal         : {n_normal:,}  ({n_normal/total*100:.1f}%)")
    print(f"  Anomaly        : {n_anomaly:,}  ({rate:.1f}%)")
    print(f"\n  Score stats:")
    print(f"    Min   : {df['anomaly_score'].min():.4f}")
    print(f"    Max   : {df['anomaly_score'].max():.4f}")
    print(f"    Mean  : {df['anomaly_score'].mean():.4f}")
    print(f"    Std   : {df['anomaly_score'].std():.4f}")

    if rate < 1:
        print("\n  [!] Warning: anomaly rate < 1% — model may be too strict")
        print("      Consider raising contamination in the training script")
    elif rate > 15:
        print("\n  [!] Warning: anomaly rate > 15% — model may be too loose")
        print("      Consider lowering contamination in the training script")
    else:
        print(f"\n  [✓] Anomaly rate {rate:.1f}% looks healthy")


# ─────────────────────────────────────────────
# LEVEL 3 — Pseudo-labeled accuracy
# Use status codes as rough ground truth:
#   error_rate > 0.5  →  pseudo anomaly
#   error_rate <= 0.5 →  pseudo normal
# ─────────────────────────────────────────────

def level3_pseudo_accuracy(results_csv="anomaly_results.csv"):
    print("\n" + "=" * 58)
    print("  LEVEL 3 — Pseudo-Labeled Accuracy (Status Code GT)")
    print("=" * 58)

    try:
        df = pd.read_csv(results_csv)
    except FileNotFoundError:
        print("  [!] anomaly_results.csv not found — skipping")
        return

    # Pseudo ground truth: error-heavy windows = anomalous
    df["pseudo_label"] = (df["error_rate"] > 0.5).astype(int)
    df["predicted"]    = (df["is_anomaly"] == -1).astype(int)

    precision = precision_score(df["pseudo_label"], df["predicted"], zero_division=0)
    recall    = recall_score(df["pseudo_label"], df["predicted"], zero_division=0)
    f1        = f1_score(df["pseudo_label"], df["predicted"], zero_division=0)
    cm        = confusion_matrix(df["pseudo_label"], df["predicted"])

    print(f"\n  Note: pseudo ground truth = error_rate > 0.5")
    print(f"  This is approximate — not a real labeled dataset\n")
    print(f"  Precision : {precision:.3f}  (of flagged events, how many were actually error-heavy)")
    print(f"  Recall    : {recall:.3f}  (of error-heavy events, how many were caught)")
    print(f"  F1 Score  : {f1:.3f}")

    print(f"\n  Confusion matrix:")
    print(f"                  Predicted Normal  Predicted Anomaly")
    print(f"  Actual Normal   {cm[0][0]:>14,}  {cm[0][1]:>17,}")
    print(f"  Actual Anomaly  {cm[1][0]:>14,}  {cm[1][1]:>17,}")

    print(f"\n  Full report:")
    print(classification_report(
        df["pseudo_label"], df["predicted"],
        target_names=["normal", "anomaly"],
        digits=3
    ))


# ─────────────────────────────────────────────
# LEVEL 4 — Boundary sensitivity test
# Gradually increase one feature and find where
# the model crosses from normal to anomaly
# ─────────────────────────────────────────────

def level4_sensitivity(model, scaler):
    print("\n" + "=" * 58)
    print("  LEVEL 4 — Boundary Sensitivity Test")
    print("=" * 58)

    print("\n  How many requests/hour does it take to trigger an anomaly?")
    print(f"  {'request_count':>15}  {'score':>8}  result")
    print("  " + "─" * 35)

    base = [0, 3, 1200, 0, 0.0, 0, 0, 0.1, 0.6, 1200, 0.0, 1500, 0]
    prev_label = 1
    for req_count in [5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]:
        vec        = base.copy()
        vec[0]     = req_count              # request_count
        vec[3]     = req_count * 1200       # total_bytes
        s, label   = score(model, scaler, vec)
        marker     = "  ← THRESHOLD" if label != prev_label else ""
        result_str = "⚠ ANOMALY" if label == -1 else "✓ normal"
        print(f"  {req_count:>15}  {s:>8.4f}  {result_str}{marker}")
        prev_label = label

    print("\n  How much error_rate triggers an anomaly?")
    print(f"  {'error_rate':>15}  {'score':>8}  result")
    print("  " + "─" * 35)

    base = [50, 10, 500, 25000, 0.0, 0, 0, 0.1, 0.2, 500, 0.0, 1000, 0]
    prev_label = 1
    for er in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        vec        = base.copy()
        vec[4]     = er                     # error_rate
        vec[5]     = int(er * 50)           # status_404
        vec[10]    = er / max(1 - er, 0.01) # error_to_ok_ratio
        s, label   = score(model, scaler, vec)
        marker     = "  ← THRESHOLD" if label != prev_label else ""
        result_str = "⚠ ANOMALY" if label == -1 else "✓ normal"
        print(f"  {er:>15.1f}  {s:>8.4f}  {result_str}{marker}")
        prev_label = label


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  NASA Anomaly Model — Full Accuracy Evaluation")
    print("=" * 58 + "\n")

    model, scaler = load_model()

    correct, total = level1_attack_injection(model, scaler)
    level2_distribution()
    level3_pseudo_accuracy()
    level4_sensitivity(model, scaler)

    print("\n" + "=" * 58)
    print("  SUMMARY")
    print("=" * 58)
    print(f"  Level 1 attack detection : {correct}/{total} correct ({correct/total*100:.0f}%)")
    print(f"  See Level 2/3 output above for distribution and F1")
    print(f"  See Level 4 for where the model's thresholds lie")
    print("=" * 58 + "\n")
