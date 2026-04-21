from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianProxy:
    key: tuple[int, int, int]
    feature_dim: int
    num_classes: int
    position_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    feature_sum: np.ndarray | None = None
    logit_sum: np.ndarray | None = None
    uncertainty_sum: float = 0.0
    language_similarity_sum: float = 0.0
    count: int = 0
    seen_views: set[int] = field(default_factory=set)
    positive_logit_mean: float = 0.0
    positive_logit_m2: float = 0.0
    language_mean: float = 0.0
    language_m2: float = 0.0

    def __post_init__(self) -> None:
        if self.feature_sum is None:
            self.feature_sum = np.zeros(self.feature_dim, dtype=np.float64)
        if self.logit_sum is None:
            self.logit_sum = np.zeros(self.num_classes, dtype=np.float64)

    def _welford_update(self, value: float, mean: float, m2: float, count: int) -> tuple[float, float]:
        delta = value - mean
        mean += delta / max(count, 1)
        delta2 = value - mean
        m2 += delta * delta2
        return mean, m2

    def update(
        self,
        point: np.ndarray,
        color: np.ndarray,
        feature: np.ndarray,
        logit: np.ndarray,
        uncertainty: float,
        language_similarity: float,
        view_id: int,
    ) -> None:
        self.count += 1
        self.position_sum += point
        self.color_sum += color
        self.feature_sum += feature
        self.logit_sum += logit
        self.uncertainty_sum += float(uncertainty)
        self.language_similarity_sum += float(language_similarity)
        self.seen_views.add(view_id)

        self.positive_logit_mean, self.positive_logit_m2 = self._welford_update(
            float(logit[-1]),
            self.positive_logit_mean,
            self.positive_logit_m2,
            self.count,
        )
        self.language_mean, self.language_m2 = self._welford_update(
            float(language_similarity),
            self.language_mean,
            self.language_m2,
            self.count,
        )

    @property
    def reliability(self) -> float:
        if self.count < 2:
            return 0.5
        logit_var = self.positive_logit_m2 / max(self.count - 1, 1)
        lang_var = self.language_m2 / max(self.count - 1, 1)
        return float(1.0 / (1.0 + np.sqrt(logit_var + 0.25 * lang_var)))


class GaussianState:
    def __init__(self, voxel_size: float) -> None:
        self.voxel_size = float(voxel_size)
        self.proxies: dict[tuple[int, int, int], GaussianProxy] = {}
        self.feature_dim = 0
        self.num_classes = 2

    def _key_from_point(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(point / self.voxel_size).astype(np.int32).tolist())

    def update(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        features: np.ndarray,
        logits: np.ndarray,
        uncertainty: np.ndarray,
        language_similarity: np.ndarray,
        view_id: int,
    ) -> None:
        if len(points) == 0:
            return

        self.feature_dim = int(features.shape[1])
        self.num_classes = int(logits.shape[1])

        for index in range(len(points)):
            key = self._key_from_point(points[index])
            proxy = self.proxies.get(key)
            if proxy is None:
                proxy = GaussianProxy(key=key, feature_dim=self.feature_dim, num_classes=self.num_classes)
                self.proxies[key] = proxy
            proxy.update(
                point=points[index],
                color=colors[index],
                feature=features[index],
                logit=logits[index],
                uncertainty=float(uncertainty[index]),
                language_similarity=float(language_similarity[index]),
                view_id=view_id,
            )

    def to_arrays(self) -> dict[str, np.ndarray]:
        if not self.proxies:
            return {
                "positions": np.empty((0, 3), dtype=np.float32),
                "colors": np.empty((0, 3), dtype=np.float32),
                "features": np.empty((0, self.feature_dim), dtype=np.float32),
                "logits": np.empty((0, self.num_classes), dtype=np.float32),
                "uncertainty": np.empty((0,), dtype=np.float32),
                "language_similarity": np.empty((0,), dtype=np.float32),
                "reliability": np.empty((0,), dtype=np.float32),
                "view_count": np.empty((0,), dtype=np.float32),
            }

        positions = []
        colors = []
        features = []
        logits = []
        uncertainty = []
        language_similarity = []
        reliability = []
        view_count = []

        for proxy in self.proxies.values():
            positions.append(proxy.position_sum / proxy.count)
            colors.append(proxy.color_sum / proxy.count)
            features.append(proxy.feature_sum / proxy.count)
            logits.append(proxy.logit_sum / proxy.count)
            uncertainty.append(proxy.uncertainty_sum / proxy.count)
            language_similarity.append(proxy.language_similarity_sum / proxy.count)
            reliability.append(proxy.reliability)
            view_count.append(len(proxy.seen_views))

        return {
            "positions": np.asarray(positions, dtype=np.float32),
            "colors": np.asarray(colors, dtype=np.float32),
            "features": np.asarray(features, dtype=np.float32),
            "logits": np.asarray(logits, dtype=np.float32),
            "uncertainty": np.asarray(uncertainty, dtype=np.float32),
            "language_similarity": np.asarray(language_similarity, dtype=np.float32),
            "reliability": np.asarray(reliability, dtype=np.float32),
            "view_count": np.asarray(view_count, dtype=np.float32),
        }
