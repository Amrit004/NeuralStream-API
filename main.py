"""
NeuralStream API
Feature store management, model registry, and real-time serving endpoints.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging

from feature_store.store import FeatureStore, FeatureView, Feature, FeatureVector
from model_registry.registry import ModelRegistry, ModelMetadata
from serving.engine import ModelServingEngine, PredictionRequest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NeuralStream starting...")
    app.state.feature_store = FeatureStore()
    await app.state.feature_store.connect()

    app.state.model_registry = ModelRegistry()
    app.state.serving_engine = ModelServingEngine(
        app.state.feature_store, app.state.model_registry
    )

    # Register sample feature views
    app.state.feature_store.register_view(FeatureView(
        name="user_features",
        entities=["user_id"],
        features=[
            Feature("age", "float"),
            Feature("account_age_days", "float"),
            Feature("total_spend", "float"),
            Feature("num_orders", "int"),
            Feature("avg_order_value", "float"),
            Feature("days_since_last_order", "float"),
        ],
        source="users_db",
        ttl_seconds=3600,
    ))

    app.state.feature_store.register_view(FeatureView(
        name="product_features",
        entities=["product_id"],
        features=[
            Feature("price", "float"),
            Feature("category_id", "int"),
            Feature("avg_rating", "float"),
            Feature("num_reviews", "int"),
            Feature("days_since_listed", "float"),
        ],
        source="products_db",
        ttl_seconds=86400,
    ))

    logger.info("NeuralStream ready.")
    yield


app = FastAPI(
    title="NeuralStream API",
    description="Real-Time ML Feature Store & Model Serving Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Feature Store endpoints ──────────────────────────────────────────────────

class WriteFeatureRequest(BaseModel):
    entity_id: str
    feature_view: str
    values: Dict[str, Any]


class ReadFeatureBatchRequest(BaseModel):
    feature_view: str
    entity_ids: List[str]


@app.post("/features/write")
async def write_features(req: WriteFeatureRequest):
    from datetime import datetime
    vector = FeatureVector(
        entity_id=req.entity_id,
        feature_view=req.feature_view,
        values=req.values,
    )
    await app.state.feature_store.write_online(vector)
    return {"status": "written", "entity_id": req.entity_id, "feature_view": req.feature_view}


@app.get("/features/{feature_view}/{entity_id}")
async def get_features(feature_view: str, entity_id: str):
    vector = await app.state.feature_store.read_online(feature_view, entity_id)
    if vector is None:
        raise HTTPException(status_code=404, detail="Features not found")
    return {"entity_id": entity_id, "feature_view": feature_view, "values": vector.values, "timestamp": str(vector.timestamp)}


@app.post("/features/batch")
async def get_features_batch(req: ReadFeatureBatchRequest):
    results = await app.state.feature_store.read_online_batch(req.feature_view, req.entity_ids)
    return {
        "feature_view": req.feature_view,
        "results": {
            eid: {"values": v.values, "timestamp": str(v.timestamp)} if v else None
            for eid, v in results.items()
        }
    }


@app.get("/features/views")
async def list_feature_views():
    return {"views": await app.state.feature_store.list_views()}


# ─── Model Registry endpoints ─────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    return {"models": app.state.model_registry.list_models()}


@app.get("/models/{name}/lineage")
async def model_lineage(name: str):
    lineage = app.state.model_registry.get_lineage(name)
    if not lineage:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"model": name, "lineage": lineage}


@app.post("/models/{name}/promote/{version}")
async def promote_model(name: str, version: str):
    app.state.model_registry.promote(name, version)
    return {"status": "promoted", "model": name, "version": version}


class ABSplitRequest(BaseModel):
    splits: Dict[str, float]  # {version: traffic_pct}


@app.post("/models/{name}/ab-split")
async def set_ab_split(name: str, req: ABSplitRequest):
    try:
        app.state.model_registry.set_ab_split(name, req.splits)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "split_configured", "model": name, "splits": req.splits}


# ─── Serving endpoints ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    entity_id: str
    feature_view: Optional[str] = None
    extra_features: Dict[str, Any] = {}
    return_features: bool = False


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]


@app.post("/predict/{model_name}")
async def predict(model_name: str, req: PredictRequest):
    """Real-time prediction with automatic feature retrieval."""
    try:
        result = await app.state.serving_engine.predict(
            PredictionRequest(
                model_name=model_name,
                entity_id=req.entity_id,
                feature_view=req.feature_view,
                extra_features=req.extra_features,
                return_features=req.return_features,
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "model": result.model_name,
        "version": result.model_version,
        "entity_id": result.entity_id,
        "prediction": result.prediction,
        "probability": result.probability,
        "latency_ms": result.latency_ms,
        "features": result.features_used,
        "timestamp": result.timestamp,
    }


@app.post("/predict/{model_name}/batch")
async def batch_predict(model_name: str, req: BatchPredictRequest):
    """Batch prediction for multiple entities."""
    requests = [
        PredictionRequest(
            model_name=model_name,
            entity_id=r.entity_id,
            feature_view=r.feature_view,
            extra_features=r.extra_features,
        )
        for r in req.requests
    ]
    results = await app.state.serving_engine.predict_batch(requests)
    return {
        "model": model_name,
        "predictions": [
            {"entity_id": r.entity_id, "prediction": r.prediction,
             "probability": r.probability, "latency_ms": r.latency_ms}
            for r in results
        ]
    }


# ─── Monitoring endpoints ─────────────────────────────────────────────────────

@app.get("/monitoring/drift")
async def drift_report():
    alerts = app.state.serving_engine.get_drift_report()
    return {
        "drift_alerts": alerts,
        "total_alerts": len(alerts),
        "status": "degraded" if alerts else "healthy",
    }


@app.get("/monitoring/stats")
async def prediction_stats():
    return app.state.serving_engine.get_prediction_stats()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "neuralstream"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
