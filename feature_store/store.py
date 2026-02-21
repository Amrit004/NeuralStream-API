"""
NeuralStream — Real-Time ML Feature Store & Model Serving Platform
Bridges data engineering and ML: compute, store, serve, and monitor features + models.
"""
import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Feature:
    name: str
    dtype: str          # float, int, string, embedding, list
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: int = 1


@dataclass
class FeatureView:
    """A logical grouping of features from a single data source."""
    name: str
    entities: List[str]           # join keys, e.g. ["user_id", "product_id"]
    features: List[Feature]
    source: str                   # data source name
    ttl_seconds: int = 86400      # how long features stay fresh
    online: bool = True
    offline: bool = True


@dataclass
class FeatureVector:
    """A concrete set of feature values for one entity."""
    entity_id: str
    feature_view: str
    values: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1


class FeatureTransform(ABC):
    """Base class for feature transformations."""

    @abstractmethod
    def fit(self, data: List[Dict]) -> "FeatureTransform":
        pass

    @abstractmethod
    def transform(self, value: Any) -> Any:
        pass

    def fit_transform(self, data: List[Dict], key: str) -> List[Any]:
        self.fit([{key: d} for d in data])
        return [self.transform(d) for d in data]


class StandardScaler(FeatureTransform):
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data: List[Dict]) -> "StandardScaler":
        values = [list(d.values())[0] for d in data if list(d.values())[0] is not None]
        self.mean = float(np.mean(values))
        self.std = float(np.std(values)) or 1.0
        return self

    def transform(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        return (float(value) - self.mean) / self.std


class BucketTransform(FeatureTransform):
    def __init__(self, boundaries: List[float]):
        self.boundaries = sorted(boundaries)

    def fit(self, data: List[Dict]) -> "BucketTransform":
        return self  # pre-defined boundaries

    def transform(self, value: Any) -> int:
        if value is None:
            return 0
        for i, boundary in enumerate(self.boundaries):
            if float(value) < boundary:
                return i
        return len(self.boundaries)


class HashEmbedding(FeatureTransform):
    """Hash categorical features into dense embeddings."""
    def __init__(self, dim: int = 16, vocab_size: int = 10000):
        self.dim = dim
        self.vocab_size = vocab_size
        self._cache: Dict[str, np.ndarray] = {}

    def fit(self, data: List[Dict]) -> "HashEmbedding":
        return self

    def transform(self, value: Any) -> List[float]:
        key = str(value)
        if key not in self._cache:
            # Deterministic hash embedding
            seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % self.vocab_size
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-10
            self._cache[key] = vec
        return self._cache[key].tolist()


class FeatureStore:
    """
    Central feature registry and serving layer.
    Manages online (low-latency Redis) and offline (historical) feature storage.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/4"):
        self.redis_url = redis_url
        self._redis = None
        self._views: Dict[str, FeatureView] = {}
        self._transforms: Dict[str, Dict[str, FeatureTransform]] = {}

    async def connect(self):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        logger.info("Feature store connected to Redis")

    def register_view(self, view: FeatureView) -> "FeatureStore":
        self._views[view.name] = view
        logger.info(f"Registered feature view: {view.name} ({len(view.features)} features)")
        return self

    def register_transform(self, view_name: str, feature_name: str, transform: FeatureTransform):
        if view_name not in self._transforms:
            self._transforms[view_name] = {}
        self._transforms[view_name][feature_name] = transform

    async def write_online(self, vector: FeatureVector):
        """Write features to online store (Redis) for low-latency serving."""
        if self._redis is None:
            return
        key = f"fs:{vector.feature_view}:{vector.entity_id}"
        ttl = self._views.get(vector.feature_view, FeatureView("", [], [], "")).ttl_seconds

        # Apply transforms
        transformed = self._apply_transforms(vector.feature_view, vector.values)
        payload = json.dumps({"values": transformed, "ts": vector.timestamp.isoformat()})
        await self._redis.setex(key, ttl, payload)

    async def read_online(self, feature_view: str, entity_id: str) -> Optional[FeatureVector]:
        """Read features from online store — p99 < 5ms."""
        if self._redis is None:
            return None
        key = f"fs:{feature_view}:{entity_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        parsed = json.loads(data)
        return FeatureVector(
            entity_id=entity_id,
            feature_view=feature_view,
            values=parsed["values"],
            timestamp=datetime.fromisoformat(parsed["ts"]),
        )

    async def read_online_batch(
        self, feature_view: str, entity_ids: List[str]
    ) -> Dict[str, Optional[FeatureVector]]:
        """Read multiple entities in one Redis pipeline call."""
        if self._redis is None:
            return {eid: None for eid in entity_ids}
        keys = [f"fs:{feature_view}:{eid}" for eid in entity_ids]
        pipe = self._redis.pipeline()
        for key in keys:
            pipe.get(key)
        results = await pipe.execute()

        output = {}
        for eid, data in zip(entity_ids, results):
            if data:
                parsed = json.loads(data)
                output[eid] = FeatureVector(
                    entity_id=eid,
                    feature_view=feature_view,
                    values=parsed["values"],
                    timestamp=datetime.fromisoformat(parsed["ts"]),
                )
            else:
                output[eid] = None
        return output

    def _apply_transforms(self, view_name: str, values: Dict[str, Any]) -> Dict[str, Any]:
        transforms = self._transforms.get(view_name, {})
        result = {}
        for k, v in values.items():
            if k in transforms:
                try:
                    result[k] = transforms[k].transform(v)
                except Exception as e:
                    logger.warning(f"Transform failed for {view_name}.{k}: {e}")
                    result[k] = v
            else:
                result[k] = v
        return result

    def get_feature_schema(self, view_name: str) -> Optional[Dict]:
        view = self._views.get(view_name)
        if not view:
            return None
        return {
            "name": view.name,
            "entities": view.entities,
            "features": [{"name": f.name, "dtype": f.dtype, "description": f.description} for f in view.features],
            "ttl_seconds": view.ttl_seconds,
        }

    async def list_views(self) -> List[Dict]:
        return [self.get_feature_schema(name) for name in self._views]
