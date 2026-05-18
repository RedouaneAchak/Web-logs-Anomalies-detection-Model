# Web-logs-Anomalies-detection-Model
this is a model (isolated forest)  that receives a normalized  1 Ip per hour window of logs and detects wither it's an anomaly or not
# Web Log Anomaly Detection Model
> Part of a multi-source AI-powered SIEM built on the ELK stack.  
> Detects anomalous behavior in web server access logs using unsupervised machine learning.

---

## Overview

This module is the **web log component** of a larger SIEM (Security Information and Event Management) system. It uses an **Isolation Forest** model trained on real NASA HTTP access logs to detect statistical anomalies in web traffic — without requiring any labeled attack data.

Unlike supervised models that only catch known attack patterns, this approach detects **novel and unknown threats** by learning what normal traffic looks like and flagging anything that deviates significantly.

The model is designed to run alongside the ELK stack (Elasticsearch, Logstash, Kibana) and feeds anomaly scores into Kibana alert rules, triggering popup notifications when suspicious activity is detected.

---

## How It Works

### Core idea
The model aggregates raw log events into **per-IP, per-hour feature windows**, then scores each window. An IP that suddenly sends 500 requests with a 90% error rate in one hour looks very different from normal browsing behavior — the model assigns it a low anomaly score and flags it.

### Algorithm — Isolation Forest
Isolation Forest works by randomly partitioning the feature space with decision trees. Normal points require many splits to isolate because they cluster together. Anomalous points are rare and unusual — they get isolated in very few splits. The fewer splits needed, the more anomalous the point.

```
Normal point  →  hard to isolate  →  high score  →  ✓ normal
Anomaly       →  easy to isolate  →  low score   →  ⚠ flagged
```

### Detection pipeline

```
Apache access.log
      ↓
Parse raw log lines (regex)
      ↓
Aggregate features per IP per hour
      ↓
Scale features (StandardScaler)
      ↓
Isolation Forest scores each window
      ↓
anomaly_score + is_anomaly flag
      ↓
Elasticsearch → Kibana alert
```

---

## Dataset

### Training data — NASA HTTP Access Log (July 1995)
- **Source:** NASA Kennedy Space Center web server
- **Size:** ~1.8 million HTTP requests
- **Format:** Apache Combined Log Format
- **Download:** `ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz`
- **Why this dataset:** Real-world traffic with a natural mix of normal requests (200s) and errors (4xx/5xx). Widely cited in academic literature. The script downloads it automatically on first run.

### Log format
Each raw line looks like this:
```
199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
```

| Field | Example | Description |
|---|---|---|
| Host/IP | `199.72.81.55` | Client IP or hostname |
| Timestamp | `01/Jul/1995:00:00:01` | Request time |
| Method | `GET` | HTTP method |
| URL | `/history/apollo/` | Requested path |
| Status | `200` | HTTP response code |
| Bytes | `6245` | Response size in bytes |

---

## Log Normalization

Raw log lines are unstructured text. Before feeding them to the model, each line is parsed and normalized into structured fields using a regex pattern matching the Apache Combined Log Format.

After parsing, events are **aggregated into 1-hour windows per IP**. This is the key normalization step — instead of scoring individual requests, the model scores behavioral patterns over time.

### Features engineered per IP per hour

| Feature | Description | Attack signal |
|---|---|---|
| `request_count` | Total requests in the window | DDoS, flooding |
| `unique_urls` | Distinct URLs requested | Directory scanning |
| `avg_bytes` | Average response size | Baseline traffic |
| `total_bytes` | Total bytes transferred | Data exfiltration |
| `max_bytes` | Largest single response | Exfiltration spike |
| `large_req_count` | Requests with response > 50KB | Bulk data transfer |
| `error_rate` | Ratio of 4xx/5xx responses | Scanning, probing |
| `status_404` | Count of Not Found responses | Path enumeration |
| `status_500` | Count of Server Error responses | Attack-triggered errors |
| `post_rate` | Ratio of POST requests | Brute force, injection |
| `url_diversity` | unique_urls / request_count | Scanning breadth |
| `bytes_per_req` | total_bytes / request_count | Transfer efficiency |
| `error_to_ok_ratio` | 404s / 200s | How many fail vs succeed |

All features are then standardized using `StandardScaler` (zero mean, unit variance) before being passed to the model.

---

## Files

```
.
├── nasa_anomaly_detection.py   Training pipeline
├── test_model.py               Model evaluation and accuracy tests
├── scoring_service.py          FastAPI service for live scoring (ELK integration)
├── anomaly_model.pkl           Trained Isolation Forest model
├── scaler.pkl                  Fitted StandardScaler
├── anomaly_results.csv         All scored IP/hour windows
├── flagged_anomalies.csv       Only the flagged anomalous windows
└── NASA_access_log_Jul95.gz    Raw dataset (downloaded automatically)
```

### `nasa_anomaly_detection.py`
The main training script. Run this to:
1. Download the NASA dataset automatically
2. Parse 1.8M raw log lines
3. Engineer 13 features per IP/hour window
4. Train the Isolation Forest model
5. Score all windows and export results
6. Save the model and scaler as `.pkl` files

```bash
python nasa_anomaly_detection.py
```

### `test_model.py`
Four-level evaluation suite to validate model quality:

- **Level 1** — Injects 18 hand-crafted attack and normal vectors and checks if the model correctly classifies each one. Includes a combo rule for exfiltration detection.
- **Level 2** — Checks that the anomaly flagging rate is within a healthy range (1–15%).
- **Level 3** — Uses HTTP status codes as pseudo-labels to compute precision, recall, and F1 score.
- **Level 4** — Gradually increases individual features to find the model's detection thresholds.

```bash
python test_model.py
```

### `scoring_service.py`
A FastAPI microservice that loads the trained model and exposes a `/score` endpoint. Logstash calls this for every incoming log event to get an anomaly score in real time. Handles all three log types (SSH, web, firewall) by routing to the correct model based on the `log_type` field.

```bash
uvicorn scoring_service:app --host 0.0.0.0 --port 8000
```

### `anomaly_model.pkl`
The serialized trained Isolation Forest model. Produced by `nasa_anomaly_detection.py`. Loaded by `scoring_service.py` and `test_model.py` at runtime. Do not edit manually.

### `scaler.pkl`
The fitted StandardScaler that was used during training. Must be applied to every new feature vector before scoring — using the raw unscaled features will produce incorrect results. Always kept in sync with `anomaly_model.pkl`.

### `anomaly_results.csv`
All IP/hour windows from the NASA dataset with their anomaly scores and labels. Columns include all 13 features plus `anomaly_score`, `is_anomaly` (-1 or 1), and `label` (ANOMALY or normal).

### `flagged_anomalies.csv`
Subset of `anomaly_results.csv` containing only the rows where `is_anomaly == -1`. Sorted by anomaly score ascending (most suspicious first). Use this to inspect what the model flagged.

---

## Quick Start

### 1. Install dependencies
```bash
pip install pandas scikit-learn joblib numpy fastapi uvicorn shap
```

### 2. Train the model
```bash
python nasa_anomaly_detection.py
```
This downloads the dataset automatically (~20MB), trains in ~5 minutes, and produces `anomaly_model.pkl`, `scaler.pkl`, `anomaly_results.csv`, and `flagged_anomalies.csv`.

### 3. Evaluate the model
```bash
python test_model.py
```

### 4. Start the scoring service
```bash
uvicorn scoring_service:app --host 0.0.0.0 --port 8000
```

### 5. Test the scoring endpoint
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "log_type": "web",
    "request_count": 450,
    "error_rate": 0.87,
    "status_404": 38,
    "unique_urls": 12,
    "avg_bytes": 320,
    "total_bytes": 144000,
    "max_bytes": 800,
    "large_req_count": 0,
    "post_rate": 0.05,
    "url_diversity": 0.027,
    "bytes_per_req": 320,
    "error_to_ok_ratio": 6.7
  }'
```

Expected response:
```json
{
  "log_type": "web",
  "is_anomaly": true,
  "anomaly_score": -0.3412,
  "label": "ANOMALY"
}
```

---

## ELK Stack Integration

This model is designed to plug into a Logstash → Elasticsearch → Kibana pipeline:

1. **Logstash** parses incoming Apache access logs and calls `POST /score` on the scoring service
2. The scoring service returns `is_anomaly` and `anomaly_score`
3. Logstash appends these fields to the event and indexes it into Elasticsearch
4. A **Kibana alert rule** queries for `is_anomaly: true` every minute and fires a popup notification

For full integration details see the parent SIEM repository.

---

## Model Configuration

| Parameter | Value | Description |
|---|---|---|
| Algorithm | Isolation Forest | Unsupervised anomaly detection |
| `n_estimators` | 200 | Number of isolation trees |
| `contamination` | 0.05 | Expected anomaly rate (5%) |
| `random_state` | 42 | Reproducibility seed |
| Time window | 1 hour | Feature aggregation window |
| Features | 13 | See feature table above |

---

## Why Unsupervised?

This model requires **no labeled attack data** to train. This is intentional:

- Real attack labels are expensive and rare to obtain
- Labeled datasets only cover **known** attack types
- Unsupervised models can detect **novel attacks** that have never been seen before
- The model adapts to whatever traffic pattern is "normal" for your server

The trade-off is that unsupervised models can miss subtle attacks that look statistically similar to normal traffic (e.g. low-and-slow attacks). This is why this model runs alongside **Sigma rules** in the SIEM — Sigma catches known attack signatures while the ML model catches statistical outliers. Together they provide double-layer coverage.

---

## Limitations

- Trained on 1995 NASA traffic — the baseline "normal" reflects that era's patterns. For production use, retrain on your own server's traffic.
- Data exfiltration detection is weaker than other attack types due to training data bias (1995 responses were naturally large). A combo rule in `test_model.py` partially compensates for this.
- The model scores aggregated windows, not individual requests. A single malicious request inside a normal hour window may not be flagged.
- No ground truth labels — accuracy is measured using pseudo-labels (status codes) and synthetic test vectors, not real verified attack data.

---

## Part of a larger SIEM

This web model is one of three models in the project:

| Component | Log source | Teammate |
|---|---|---|
| **Web model** (this repo) | `/var/log/apache2/access.log` | You |
| SSH model | `/var/log/auth.log` | Teammate 1 |
| Firewall model | `/var/log/ufw.log` | Teammate 2 |

All three models are served by a single unified `scoring_service.py` and feed into a shared Kibana dashboard with Sigma rules for rule-based double detection.
