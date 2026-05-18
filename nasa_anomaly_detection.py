"""
NASA HTTP Log - Anomaly Detection Pipeline
============================================
Full pipeline: download → parse → feature engineering → 
Isolation Forest training → scoring → export for Elasticsearch

Requirements:
    pip install pandas scikit-learn requests joblib

Usage:
    python nasa_anomaly_detection.py

Output:
    - anomaly_results.csv     : scored records, ready to review
    - anomaly_model.pkl       : trained Isolation Forest model
    - scaler.pkl              : fitted StandardScaler
    - flagged_anomalies.csv   : only the flagged rows (is_anomaly == -1)
"""

from pyexpat import features
import re
import gzip
import os
import joblib
import urllib.request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LOG_URL       = "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
LOG_GZ        = "NASA_access_log_Jul95.gz"
LOG_FILE      = "NASA_access_log_Jul95.txt"
RESULTS_CSV   = "anomaly_results.csv"
FLAGGED_CSV   = "flagged_anomalies.csv"
MODEL_FILE    = "anomaly_model.pkl"
SCALER_FILE   = "scaler.pkl"

TIME_WINDOW   = "1H"        # Aggregate features per IP per hour
CONTAMINATION = 0.05        # Expected anomaly rate (1%)
N_ESTIMATORS  = 200         # Isolation Forest trees


# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────

def download_dataset():
    if os.path.exists(LOG_FILE):
        print(f"[✓] Log file already exists: {LOG_FILE}")
        return
    if not os.path.exists(LOG_GZ):
        print(f"[→] Downloading NASA HTTP log dataset...")
        urllib.request.urlretrieve(LOG_URL, LOG_GZ)
        print(f"[✓] Downloaded: {LOG_GZ}")
    print(f"[→] Decompressing...")
    with gzip.open(LOG_GZ, "rb") as f_in:
        with open(LOG_FILE, "wb") as f_out:
            f_out.write(f_in.read())
    print(f"[✓] Decompressed: {LOG_FILE}")


# ─────────────────────────────────────────────
# STEP 2 — PARSE RAW LOGS
# ─────────────────────────────────────────────

# Apache Combined Log Format pattern
LOG_PATTERN = re.compile(
    r'(?P<host>\S+)'           # client host/IP
    r' - -'                    # ident / authuser (always - -)
    r' \[(?P<time>[^\]]+)\]'   # timestamp
    r' "(?P<method>\S+)'       # HTTP method
    r' (?P<url>\S+)'           # requested URL
    r' \S+"'                   # HTTP version
    r' (?P<status>\d{3})'      # status code
    r' (?P<bytes>\S+)'         # bytes transferred
)

def parse_logs(filepath):
    print(f"[→] Parsing log file: {filepath}")
    records = []
    skipped = 0

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.match(line.strip())
            if not m:
                skipped += 1
                continue
            records.append({
                "host"  : m.group("host"),
                "time"  : m.group("time"),
                "method": m.group("method"),
                "url"   : m.group("url"),
                "status": int(m.group("status")),
                "bytes" : 0 if m.group("bytes") == "-" else int(m.group("bytes")),
            })

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    df["hour"] = df["time"].dt.floor(TIME_WINDOW)

    print(f"[✓] Parsed {len(df):,} entries  |  Skipped {skipped:,} malformed lines")
    return df


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    print("[→] Engineering features per IP per hour window...")

    features = df.groupby(["host", "hour"]).agg(
        request_count = ("url",    "count"),
        unique_urls   = ("url",    "nunique"),
        avg_bytes     = ("bytes",  "mean"),
        total_bytes   = ("bytes",  "sum"),
        error_rate    = ("status", lambda x: (x >= 400).mean()),
        status_404    = ("status", lambda x: (x == 404).sum()),
        status_500    = ("status", lambda x: (x == 500).sum()),
        status_200    = ("status", lambda x: (x == 200).sum()),
        post_rate     = ("method", lambda x: (x == "POST").mean()),
        get_rate      = ("method", lambda x: (x == "GET").mean()),
    ).reset_index()

    # Derived features
    features["url_diversity"]    = features["unique_urls"] / features["request_count"].clip(lower=1)
    features["bytes_per_req"]    = features["total_bytes"] / features["request_count"].clip(lower=1)
    features["error_to_ok_ratio"]= features["status_404"] / features["status_200"].clip(lower=1)
    # Max bytes in a single request — flags exfiltration better
    features["max_bytes"] = df.groupby(["host","hour"])["bytes"].max().values
    # Requests with bytes > 50k — direct exfil indicator  
    features["large_req_count"] = df.groupby(["host","hour"])["bytes"].apply(
    lambda x: (x > 50000).sum()
).values

    print(f"[✓] Feature table: {len(features):,} rows (IP × hour combinations)")
    return features


# ─────────────────────────────────────────────
# STEP 4 — TRAIN ISOLATION FOREST
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

def train_model(features):
    print(f"[→] Training Isolation Forest  (contamination={CONTAMINATION}, trees={N_ESTIMATORS})...")

    X = features[FEATURE_COLS].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    print("[✓] Model trained")
    return model, scaler, X_scaled


# ─────────────────────────────────────────────
# STEP 5 — SCORE + EXPORT
# ─────────────────────────────────────────────

def score_and_export(features, model, scaler, X_scaled):
    print("[→] Scoring all records...")

    # anomaly_score: lower (more negative) = more anomalous
    # is_anomaly:    -1 = anomaly,  1 = normal
    features["anomaly_score"] = model.decision_function(X_scaled)
    features["is_anomaly"]    = model.predict(X_scaled)

    # Human-readable flag
    features["label"] = features["is_anomaly"].map({1: "normal", -1: "ANOMALY"})

    # Sort: most anomalous first
    results = features.sort_values("anomaly_score", ascending=True)
    flagged = results[results["is_anomaly"] == -1]

    # Save
    results.to_csv(RESULTS_CSV, index=False)
    flagged.to_csv(FLAGGED_CSV, index=False)

    total    = len(results)
    n_flagged= len(flagged)

    print(f"[✓] Scored {total:,} records")
    print(f"[✓] Flagged {n_flagged:,} anomalies ({n_flagged/total*100:.1f}%)")
    print(f"[✓] Results saved → {RESULTS_CSV}")
    print(f"[✓] Flagged only  → {FLAGGED_CSV}")

    # Print top 10 most anomalous
    print("\n── Top 10 Most Anomalous IP/Hour combinations ──")
    cols = ["host", "hour", "request_count", "error_rate", "status_404", "unique_urls", "anomaly_score"]
    print(flagged[cols].head(10).to_string(index=False))

    return results, flagged


# ─────────────────────────────────────────────
# STEP 6 — SAVE MODEL
# ─────────────────────────────────────────────

def save_model(model, scaler):
    joblib.dump(model,  MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n[✓] Model saved  → {MODEL_FILE}")
    print(f"[✓] Scaler saved → {SCALER_FILE}")
    print("    (Load these in your Logstash scoring service)")


# ─────────────────────────────────────────────
# SCORING SERVICE SNIPPET (for reference)
# ─────────────────────────────────────────────

SCORING_SNIPPET = '''
# ── scoring_service.py ──────────────────────────────────────────────
# Drop this FastAPI service alongside your ELK stack.
# Logstash calls POST /score with a log event, gets back anomaly_score.
#
# Install: pip install fastapi uvicorn joblib scikit-learn
# Run:     uvicorn scoring_service:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app    = FastAPI()
model  = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLS = [
    "request_count","unique_urls","avg_bytes","total_bytes",
    "error_rate","status_404","status_500","post_rate",
    "url_diversity","bytes_per_req","error_to_ok_ratio",
]

class LogEvent(BaseModel):
    request_count: float = 0
    unique_urls: float = 0
    avg_bytes: float = 0
    total_bytes: float = 0
    error_rate: float = 0
    status_404: float = 0
    status_500: float = 0
    post_rate: float = 0
    url_diversity: float = 0
    bytes_per_req: float = 0
    error_to_ok_ratio: float = 0

@app.post("/score")
def score(event: LogEvent):
    X = np.array([[getattr(event, f) for f in FEATURE_COLS]])
    X_scaled = scaler.transform(X)
    score  = float(model.decision_function(X_scaled)[0])
    label  = int(model.predict(X_scaled)[0])          # -1 or 1
    return {
        "anomaly_score": score,
        "is_anomaly": label == -1,
        "label": "ANOMALY" if label == -1 else "normal"
    }
'''


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NASA HTTP Log — Anomaly Detection Pipeline")
    print("=" * 55)

    download_dataset()
    df       = parse_logs(LOG_FILE)
    features = engineer_features(df)
    model, scaler, X_scaled = train_model(features)
    results, flagged        = score_and_export(features, model, scaler, X_scaled)
    save_model(model, scaler)

    # Write scoring service to disk for reference
    with open("scoring_service.py", "w") as f:
        f.write(SCORING_SNIPPET)
    print("[✓] Scoring service stub → scoring_service.py")

    print("\n── Next steps ───────────────────────────────────")
    print("  1. Point Filebeat at your live Apache access.log")
    print("  2. Run scoring_service.py alongside your ELK stack")
    print("  3. Add Logstash http filter → POST features to /score")
    print("  4. Index is_anomaly field into Elasticsearch")
    print("  5. Create Kibana alert rule on is_anomaly == true")
    print("=" * 55)
