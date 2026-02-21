🚀 NeuralStream
Real-Time ML Feature Store & Model Serving Platform

NeuralStream is a production-style machine learning infrastructure platform built with FastAPI. It provides:

⚡ Real-time feature store

📦 Model registry with versioning & lineage

🎯 A/B model traffic splitting

🔮 Real-time and batch prediction endpoints

📊 Drift monitoring & prediction statistics

🏗 Architecture Overview

NeuralStream consists of three core components:

1️⃣ Feature Store

Online feature retrieval

Feature views with TTL

Batch and single-entity reads

Redis-backed storage

2️⃣ Model Registry

Model version tracking

Promotion system (staging → production)

A/B traffic splitting

Lineage tracking

3️⃣ Serving Engine

Automatic feature retrieval

Real-time predictions

Batch inference

Drift detection

Latency tracking

📁 Project Structure
neuralstream/
│
├── main.py
├── requirements.txt
├── README.md
│
├── feature_store/
│   └── store.py
│
├── model_registry/
│   └── registry.py
│
├── serving/
│   └── engine.py
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/neuralstream.git
cd neuralstream
2️⃣ Create virtual environment
python -m venv venv

Activate it:

Windows (Git Bash):

source venv/Scripts/activate

PowerShell:

venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Start Redis

If using Docker:

docker run -p 6379:6379 redis
5️⃣ Run the API
uvicorn main:app --reload

Server will run at:

http://127.0.0.1:8000

Swagger docs:

http://127.0.0.1:8000/docs
📌 Core API Endpoints
Health Check
GET /health
Feature Store

Write Features:

POST /features/write

Get Features:

GET /features/{feature_view}/{entity_id}

Batch Read:

POST /features/batch

List Feature Views:

GET /features/views
Model Registry

List Models:

GET /models

Model Lineage:

GET /models/{name}/lineage

Promote Model:

POST /models/{name}/promote/{version}

Configure A/B Split:

POST /models/{name}/ab-split
Serving

Real-time Prediction:

POST /predict/{model_name}

Batch Prediction:

POST /predict/{model_name}/batch
Monitoring

Drift Report:

GET /monitoring/drift

Prediction Stats:

GET /monitoring/stats
🧠 Example Prediction Request
POST /predict/churn_model
{
  "entity_id": "user_123",
  "feature_view": "user_features",
  "return_features": true
}

Response:

{
  "model": "churn_model",
  "version": "v1",
  "entity_id": "user_123",
  "prediction": 1,
  "probability": 0.87,
  "latency_ms": 12.3,
  "features": {...},
  "timestamp": "2026-02-21T10:15:00"
}
🛠 Technologies Used

FastAPI

Uvicorn

Redis (async)

Pydantic

NumPy

🎯 Key Features

Production-style ML infrastructure design

Clean separation of feature store, registry, and serving

Real-time inference

A/B model routing

Drift monitoring

Extensible architecture

🚀 Future Improvements

Persistent model storage

Authentication & API keys

Metrics dashboard (Prometheus/Grafana)

Kubernetes deployment

CI/CD integration

👤 Author

Amritpal Singh Kaur
