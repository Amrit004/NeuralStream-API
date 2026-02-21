"""
NeuralStream Model Serving Engine
Real-time inference with feature retrieval, prediction logging, and drift detection.
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from feature_store.store import FeatureStore
from model_registry.registry import ModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    model_name: str
    entity_id: str
    feature_view: Optional[str] = None  # override model's default view
    extra_features: Dict[str, Any] = field(default_factory=dict)
    return_features: bool = False


@dataclass
class PredictionResult:
    model_name: str
    model_version: str
    entity_id: str
    prediction: Any
    probability: Optional[float] = None
    features_used: Optional[Dict] = None
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DriftDetector:
    """
    Population Stability Index (PSI) based feature drift detection.
    Alerts when production feature distributions deviate from training baseline.
    """

    def __init__(self, window_size: int = 1000, psi_threshold: float = 0.2):
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        self._baselines: Dict[str, np.ndarray] = {}
        self._production: Dict[str, deque] = {}

    def set_baseline(self, feature_name: str, values: List[float]):
        self._baselines[feature_name] = np.array(values)
        logger.info(f"Drift baseline set for feature: {feature_name}")

    def record(self, feature_name: str, value: float):
        if feature_name not in self._production:
            self._production[feature_name] = deque(maxlen=self.window_size)
        self._production[feature_name].append(value)

    def check_drift(self, feature_name: str) -> Optional[Dict]:
        """Returns drift report if PSI > threshold, else None."""
        if feature_name not in self._baselines or feature_name not in self._production:
            return None
        if len(self._production[feature_name]) < 100:
            return None

        baseline = self._baselines[feature_name]
        production = np.array(list(self._production[feature_name]))
        psi = self._compute_psi(baseline, production)

        if psi > self.psi_threshold:
            severity = "critical" if psi > 0.5 else "warning"
            logger.warning(f"DRIFT DETECTED: {feature_name} PSI={psi:.3f} [{severity}]")
            return {
                "feature": feature_name,
                "psi": round(psi, 4),
                "severity": severity,
                "baseline_mean": round(float(np.mean(baseline)), 4),
                "production_mean": round(float(np.mean(production)), 4),
                "baseline_std": round(float(np.std(baseline)), 4),
                "production_std": round(float(np.std(production)), 4),
            }
        return None

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index = sum[(A-E) * ln(A/E)]"""
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max()) + 1e-10
        boundaries = np.linspace(min_val, max_val, bins + 1)

        expected_pcts = np.histogram(expected, bins=boundaries)[0] / len(expected)
        actual_pcts = np.histogram(actual, bins=boundaries)[0] / len(actual)

        # Avoid log(0)
        expected_pcts = np.clip(expected_pcts, 1e-4, None)
        actual_pcts = np.clip(actual_pcts, 1e-4, None)

        return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))


class ModelServingEngine:
    """
    End-to-end prediction pipeline:
    1. Retrieve features from online store
    2. Select model version (champion / A/B split)
    3. Run inference
    4. Log prediction + monitor drift
    """

    def __init__(self, feature_store: FeatureStore, model_registry: ModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.drift_detector = DriftDetector()
        self._prediction_log: deque = deque(maxlen=10000)

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        t0 = time.perf_counter()

        # 1. Select model version
        meta = self.model_registry.get_serving_version(request.model_name)
        if meta is None:
            raise ValueError(f"No model registered: {request.model_name}")

        # 2. Retrieve features
        view_name = request.feature_view or meta.feature_view
        feature_vector = await self.feature_store.read_online(view_name, request.entity_id)

        features: Dict[str, Any] = {}
        if feature_vector:
            features.update(feature_vector.values)
        features.update(request.extra_features)

        # 3. Load and run model
        model = self.model_registry.load_model(meta.name, meta.version)
        if model is None:
            raise RuntimeError(f"Failed to load model: {meta.name} v{meta.version}")

        prediction, probability = self._run_inference(model, features, meta)

        # 4. Record for drift monitoring
        for feat_name, feat_val in features.items():
            if isinstance(feat_val, (int, float)):
                self.drift_detector.record(feat_name, float(feat_val))

        latency_ms = (time.perf_counter() - t0) * 1000
        result = PredictionResult(
            model_name=meta.name,
            model_version=meta.version,
            entity_id=request.entity_id,
            prediction=prediction,
            probability=probability,
            features_used=features if request.return_features else None,
            latency_ms=round(latency_ms, 2),
        )

        self._prediction_log.append(result)
        logger.debug(f"Prediction: {request.model_name} v{meta.version} | {request.entity_id} → {prediction} ({latency_ms:.1f}ms)")
        return result

    async def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResult]:
        """Run batch predictions in parallel."""
        return await asyncio.gather(*[self.predict(r) for r in requests])

    def _run_inference(self, model: Any, features: Dict, meta: ModelMetadata):
        """Run inference, handling different frameworks."""
        feature_values = list(features.values())
        numeric = [v for v in feature_values if isinstance(v, (int, float))]
        X = np.array(numeric, dtype=np.float32).reshape(1, -1)

        try:
            if meta.framework in ("sklearn", "xgboost", "lightgbm"):
                if meta.task == "classification":
                    proba = model.predict_proba(X)[0]
                    pred = int(np.argmax(proba))
                    return pred, float(np.max(proba))
                else:
                    pred = float(model.predict(X)[0])
                    return pred, None

            elif meta.framework == "tensorflow":
                import tensorflow as tf
                pred = model(X, training=False).numpy()[0]
                if len(pred) > 1:
                    return int(np.argmax(pred)), float(np.max(pred))
                return float(pred[0]), None

            elif meta.framework == "onnx":
                input_name = model.get_inputs()[0].name
                pred = model.run(None, {input_name: X})[0][0]
                if hasattr(pred, "__len__") and len(pred) > 1:
                    return int(np.argmax(pred)), float(np.max(pred))
                return float(pred), None

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None, None

        return None, None

    def get_drift_report(self) -> List[Dict]:
        """Return drift alerts for all monitored features."""
        alerts = []
        all_features = set(self.drift_detector._production.keys())
        for feat in all_features:
            alert = self.drift_detector.check_drift(feat)
            if alert:
                alerts.append(alert)
        return alerts

    def get_prediction_stats(self) -> Dict:
        if not self._prediction_log:
            return {"total": 0}
        latencies = [p.latency_ms for p in self._prediction_log]
        return {
            "total_predictions": len(self._prediction_log),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        }
