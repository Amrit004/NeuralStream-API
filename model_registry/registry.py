"""
NeuralStream Model Registry
Version-controlled model storage with metadata, lineage tracking, and A/B routing.
"""
import json
import logging
import os
import pickle
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    name: str
    version: str
    framework: str          # sklearn, tensorflow, pytorch, xgboost, onnx
    task: str               # classification, regression, ranking, embedding
    feature_view: str       # which feature view this model consumes
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    artifact_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = ""
    description: str = ""
    is_champion: bool = False  # currently serving model
    traffic_pct: float = 100.0  # % of traffic for A/B testing


@dataclass
class ModelVersion:
    metadata: ModelMetadata
    model: Any  # the actual model object


class ModelRegistry:
    """
    Central registry for all ML models.
    Handles versioning, promotion (challenger → champion), and A/B traffic splitting.
    """

    def __init__(self, storage_dir: str = "model_registry/artifacts"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._registry: Dict[str, List[ModelMetadata]] = {}  # name -> list of versions
        self._loaded: Dict[str, Any] = {}  # "name:version" -> model object
        self._load_index()

    def register(self, model: Any, metadata: ModelMetadata) -> str:
        """Register a trained model and persist it."""
        version_id = metadata.version or self._next_version(metadata.name)
        metadata.version = version_id

        # Persist model artifact
        artifact_path = os.path.join(self.storage_dir, f"{metadata.name}_v{version_id}")
        self._save_model(model, artifact_path, metadata.framework)
        metadata.artifact_path = artifact_path

        # Update registry
        if metadata.name not in self._registry:
            self._registry[metadata.name] = []
        self._registry[metadata.name].append(metadata)

        self._save_index()
        logger.info(f"Model registered: {metadata.name} v{version_id} ({metadata.framework})")
        return version_id

    def promote(self, name: str, version: str):
        """Promote a challenger to champion (100% traffic)."""
        versions = self._registry.get(name, [])
        for meta in versions:
            if meta.version == version:
                meta.is_champion = True
                meta.traffic_pct = 100.0
            else:
                meta.is_champion = False
                meta.traffic_pct = 0.0
        self._save_index()
        logger.info(f"Model promoted: {name} v{version} is now champion")

    def set_ab_split(self, name: str, splits: Dict[str, float]):
        """Set traffic split for A/B testing. splits = {version: pct}"""
        if abs(sum(splits.values()) - 100.0) > 0.01:
            raise ValueError(f"Traffic splits must sum to 100, got {sum(splits.values())}")
        versions = self._registry.get(name, [])
        for meta in versions:
            meta.traffic_pct = splits.get(meta.version, 0.0)
            meta.is_champion = meta.traffic_pct >= 50.0
        self._save_index()
        logger.info(f"A/B split set for {name}: {splits}")

    def get_serving_version(self, name: str) -> Optional[ModelMetadata]:
        """Get the version to serve, respecting A/B traffic splits."""
        import random
        versions = [m for m in self._registry.get(name, []) if m.traffic_pct > 0]
        if not versions:
            return None
        if len(versions) == 1:
            return versions[0]
        # Weighted random selection
        weights = [m.traffic_pct for m in versions]
        return random.choices(versions, weights=weights, k=1)[0]

    def load_model(self, name: str, version: str) -> Optional[Any]:
        """Load model artifact into memory (cached)."""
        cache_key = f"{name}:{version}"
        if cache_key in self._loaded:
            return self._loaded[cache_key]

        meta = self._get_metadata(name, version)
        if meta is None or not meta.artifact_path:
            return None

        model = self._load_model_artifact(meta.artifact_path, meta.framework)
        if model is not None:
            self._loaded[cache_key] = model
        return model

    def list_models(self) -> List[Dict]:
        result = []
        for name, versions in self._registry.items():
            for meta in versions:
                result.append({
                    "name": meta.name,
                    "version": meta.version,
                    "framework": meta.framework,
                    "task": meta.task,
                    "metrics": meta.metrics,
                    "is_champion": meta.is_champion,
                    "traffic_pct": meta.traffic_pct,
                    "created_at": meta.created_at,
                })
        return result

    def get_lineage(self, name: str) -> List[Dict]:
        return [
            {"version": m.version, "created_at": m.created_at,
             "metrics": m.metrics, "is_champion": m.is_champion}
            for m in self._registry.get(name, [])
        ]

    def _get_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        for meta in self._registry.get(name, []):
            if meta.version == version:
                return meta
        return None

    def _next_version(self, name: str) -> str:
        existing = self._registry.get(name, [])
        return str(len(existing) + 1)

    def _save_model(self, model: Any, path: str, framework: str):
        if framework in ("sklearn", "xgboost", "lightgbm"):
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model, f)
        elif framework == "tensorflow":
            model.save(f"{path}.keras")
        elif framework == "pytorch":
            import torch
            torch.save(model.state_dict(), f"{path}.pt")
        elif framework == "onnx":
            with open(f"{path}.onnx", "wb") as f:
                f.write(model)
        else:
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(model, f)

    def _load_model_artifact(self, path: str, framework: str) -> Optional[Any]:
        try:
            if framework in ("sklearn", "xgboost", "lightgbm", ""):
                with open(f"{path}.pkl", "rb") as f:
                    return pickle.load(f)
            elif framework == "tensorflow":
                import tensorflow as tf
                return tf.keras.models.load_model(f"{path}.keras")
            elif framework == "onnx":
                import onnxruntime as ort
                return ort.InferenceSession(f"{path}.onnx")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
        return None

    def _save_index(self):
        index_path = os.path.join(self.storage_dir, "registry_index.json")
        data = {
            name: [
                {k: v for k, v in vars(m).items()} for m in versions
            ]
            for name, versions in self._registry.items()
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_index(self):
        index_path = os.path.join(self.storage_dir, "registry_index.json")
        if not os.path.exists(index_path):
            return
        try:
            with open(index_path) as f:
                data = json.load(f)
            for name, versions in data.items():
                self._registry[name] = [
                    ModelMetadata(**{k: v for k, v in m.items()})
                    for m in versions
                ]
        except Exception as e:
            logger.error(f"Failed to load registry index: {e}")
