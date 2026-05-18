"""
Unified Anomaly Detection Scoring Service
==========================================
One FastAPI service that routes scoring requests to the correct
model based on log_type: "ssh" | "web" | "firewall"

Requirements:
    pip install fastapi uvicorn scikit-learn joblib numpy

Run:
    uvicorn scoring_service:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /score          → score a single log event
    GET  /health         → check all models are loaded
    GET  /docs           → auto-generated API docs (FastAPI built-in)
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from contextlib import asynccontextmanager


# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# Each list must match exactly what the model
# was trained on — same columns, same order.
# ─────────────────────────────────────────────

SSH_FEATURES = [
    "failed_attempts",       # total failed logins in the window
    "unique_users_tried",    # distinct usernames attempted
    "root_login_attempts",   # attempts targeting root
    "success_rate",          # ratio of successes to total attempts
    "attempts_per_minute",   # rate of attempts
    "unique_source_ips",     # distinct IPs in window
    "off_hours_ratio",       # ratio of attempts outside 08:00-18:00
]

WEB_FEATURES = [
    "request_count",         # total requests in the window
    "unique_urls",           # distinct URLs requested
    "avg_bytes",             # average response size
    "total_bytes",           # total bytes transferred
    "error_rate",            # ratio of 4xx/5xx responses
    "status_404",            # count of 404s
    "status_500",            # count of 500s
    "post_rate",             # ratio of POST requests
    "url_diversity",         # unique_urls / request_count
    "bytes_per_req",         # total_bytes / request_count
    "error_to_ok_ratio",     # 404s / 200s
]

FIREWALL_FEATURES = [
    "blocked_count",         # total blocked packets in window
    "allowed_count",         # total allowed packets
    "unique_dest_ports",     # distinct destination ports targeted
    "unique_src_ips",        # distinct source IPs
    "block_rate",            # blocked / total packets
    "syn_count",             # SYN packets (flood indicator)
    "icmp_count",            # ICMP packets (ping flood indicator)
    "port_scan_score",       # unique ports / total packets
    "repeated_block_ips",    # IPs blocked more than once
]

FEATURE_MAP = {
    "ssh":      SSH_FEATURES,
    "web":      WEB_FEATURES,
    "firewall": FIREWALL_FEATURES,
}


# ─────────────────────────────────────────────
# MODEL REGISTRY
# Maps log_type → { model, scaler }
# Populated at startup via lifespan handler.
# ─────────────────────────────────────────────

MODEL_FILES = {
    "ssh": {
        "model":  "ssh_anomaly_model.pkl",
        "scaler": "ssh_scaler.pkl",
    },
    "web": {
        "model":  "web_anomaly_model.pkl",  # produced by nasa_anomaly_detection.py
        "scaler": "web_scaler.pkl",
    },
    "firewall": {
        "model":  "fw_anomaly_model.pkl",
        "scaler": "fw_scaler.pkl",
    },
}

models  = {}   # { "ssh": model,  "web": model,  "firewall": model  }
scalers = {}   # { "ssh": scaler, "web": scaler, "firewall": scaler }


# ─────────────────────────────────────────────
# LIFESPAN — runs once at startup and shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models into memory at startup."""
    print("\n── Loading anomaly detection models ──")
    for log_type, paths in MODEL_FILES.items():
        model_path  = paths["model"]
        scaler_path = paths["scaler"]

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models[log_type]  = joblib.load(model_path)
            scalers[log_type] = joblib.load(scaler_path)
            print(f"  [✓] {log_type:10s} model loaded  ({model_path})")
        else:
            print(f"  [!] {log_type:10s} model NOT FOUND — {model_path} missing")
            print(f"       Train the model first and place .pkl files here.")

    loaded = list(models.keys())
    print(f"\n  Ready: {loaded if loaded else 'NO MODELS LOADED'}\n")
    yield
    print("── Shutting down scoring service ──")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="SIEM Anomaly Scoring Service",
    description="Routes log events to the correct trained model based on log_type.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class SSHEvent(BaseModel):
    failed_attempts:     float = 0
    unique_users_tried:  float = 0
    root_login_attempts: float = 0
    success_rate:        float = 0
    attempts_per_minute: float = 0
    unique_source_ips:   float = 0
    off_hours_ratio:     float = 0

class WebEvent(BaseModel):
    request_count:    float = 0
    unique_urls:      float = 0
    avg_bytes:        float = 0
    total_bytes:      float = 0
    error_rate:       float = 0
    status_404:       float = 0
    status_500:       float = 0
    post_rate:        float = 0
    url_diversity:    float = 0
    bytes_per_req:    float = 0
    error_to_ok_ratio:float = 0

class FirewallEvent(BaseModel):
    blocked_count:       float = 0
    allowed_count:       float = 0
    unique_dest_ports:   float = 0
    unique_src_ips:      float = 0
    block_rate:          float = 0
    syn_count:           float = 0
    icmp_count:          float = 0
    port_scan_score:     float = 0
    repeated_block_ips:  float = 0

class ScoreRequest(BaseModel):
    log_type: Literal["ssh", "web", "firewall"] = Field(
        ..., description="Which log type this event belongs to"
    )
    # SSH fields
    failed_attempts:     float = 0
    unique_users_tried:  float = 0
    root_login_attempts: float = 0
    success_rate:        float = 0
    attempts_per_minute: float = 0
    unique_source_ips:   float = 0
    off_hours_ratio:     float = 0
    # Web fields
    request_count:       float = 0
    unique_urls:         float = 0
    avg_bytes:           float = 0
    total_bytes:         float = 0
    error_rate:          float = 0
    status_404:          float = 0
    status_500:          float = 0
    post_rate:           float = 0
    url_diversity:       float = 0
    bytes_per_req:       float = 0
    error_to_ok_ratio:   float = 0
    # Firewall fields
    blocked_count:       float = 0
    allowed_count:       float = 0
    unique_dest_ports:   float = 0
    unique_src_ips:      float = 0
    block_rate:          float = 0
    syn_count:           float = 0
    icmp_count:          float = 0
    port_scan_score:     float = 0
    repeated_block_ips:  float = 0

class ScoreResponse(BaseModel):
    log_type:      str
    is_anomaly:    bool
    anomaly_score: float
    label:         str   # "ANOMALY" or "normal"


# ─────────────────────────────────────────────
# CORE SCORING FUNCTION
# ─────────────────────────────────────────────

def score_event(log_type: str, event: ScoreRequest) -> ScoreResponse:
    """Extract the right features, scale, predict, return result."""

    if log_type not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{log_type}' is not loaded. "
                   f"Train it first and place the .pkl files in this directory."
        )

    feature_cols = FEATURE_MAP[log_type]
    model        = models[log_type]
    scaler       = scalers[log_type]

    # Extract only the features this model was trained on
    values = [getattr(event, col, 0.0) for col in feature_cols]
    X      = np.array(values, dtype=float).reshape(1, -1)

    # Scale → predict
    X_scaled      = scaler.transform(X)
    raw_score     = float(model.decision_function(X_scaled)[0])
    prediction    = int(model.predict(X_scaled)[0])   # -1 = anomaly, 1 = normal

    return ScoreResponse(
        log_type      = log_type,
        is_anomaly    = (prediction == -1),
        anomaly_score = raw_score,
        label         = "ANOMALY" if prediction == -1 else "normal",
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse)
def score(event: ScoreRequest):
    """
    Score a single log event.

    Send all feature fields relevant to your log_type.
    Fields for other log types are ignored.

    Example (web log):
    {
        "log_type": "web",
        "request_count": 450,
        "error_rate": 0.87,
        "unique_urls": 12,
        "avg_bytes": 320,
        "status_404": 38,
        "url_diversity": 0.027,
        "bytes_per_req": 320,
        "error_to_ok_ratio": 0.95
    }
    """
    return score_event(event.log_type, event)


@app.get("/health")
def health():
    """Returns which models are currently loaded and ready."""
    return {
        "status": "ok" if models else "no models loaded",
        "loaded_models": list(models.keys()),
        "missing_models": [k for k in MODEL_FILES if k not in models],
    }


@app.get("/")
def root():
    return {
        "service": "SIEM Anomaly Scoring Service",
        "endpoints": {
            "POST /score":  "Score a log event",
            "GET  /health": "Check model status",
            "GET  /docs":   "Interactive API docs",
        }
    }
