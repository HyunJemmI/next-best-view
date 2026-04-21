from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.base_model import BaseSemanticModel
from perception.language_features import ToyLanguageEncoder
from perception.uncertainty import entropy_from_logits


def resolve_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DeviceReport:
    requested: str
    resolved: str
    mps_available: bool


class GaussianSemanticWrapper(BaseSemanticModel):
    def __init__(self, config: dict) -> None:
        model_config = config["model"]
        perception_config = config["perception"]
        self.device = resolve_device(model_config["device"])
        self.device_report = DeviceReport(
            requested=str(model_config["device"]),
            resolved=str(self.device),
            mps_available=bool(torch.backends.mps.is_available()),
        )
        self.embedding_dim = int(model_config["embedding_dim"])
        self.target_radius = float(perception_config["target_radius_hint"])
        self.feature_jitter_scale = float(model_config["feature_jitter_scale"])
        self.encoder = ToyLanguageEncoder(embedding_dim=self.embedding_dim)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config["experiment"]["seed"]))
        projection = torch.randn(6, self.embedding_dim, generator=generator, dtype=torch.float32)
        projection = projection / projection.norm(dim=0, keepdim=True).clamp(min=1e-6)
        self.projection = projection.to(self.device)

    def infer(self, observation: dict) -> dict:
        points = torch.from_numpy(observation["points"]).to(self.device, dtype=torch.float32)
        colors = torch.from_numpy(observation["colors"]).to(self.device, dtype=torch.float32)
        target_center = torch.as_tensor(
            observation["target_center"],
            dtype=torch.float32,
            device=self.device,
        )

        if points.numel() == 0:
            empty = np.empty((0,), dtype=np.float32)
            return {
                "features": np.empty((0, self.embedding_dim), dtype=np.float32),
                "logits": np.empty((0, 2), dtype=np.float32),
                "predictions": np.empty((0,), dtype=np.int64),
                "uncertainty": empty,
                "language_similarity": empty,
            }

        query_embedding = self.encoder.encode(observation["query_text"], self.device)
        background_embedding = self.encoder.encode("background support surface", self.device)

        distances = torch.linalg.norm(points - target_center.unsqueeze(0), dim=1)
        objectness = torch.exp(-0.5 * (distances / max(self.target_radius, 1e-4)) ** 2)

        raw_features = torch.cat([points, colors], dim=1)
        projected = torch.tanh(raw_features @ self.projection) * self.feature_jitter_scale
        semantic_basis = (
            objectness.unsqueeze(1) * query_embedding.unsqueeze(0)
            + (1.0 - objectness).unsqueeze(1) * background_embedding.unsqueeze(0)
        )
        features = semantic_basis + projected
        features = features / features.norm(dim=1, keepdim=True).clamp(min=1e-6)

        language_similarity = torch.sum(features * query_embedding.unsqueeze(0), dim=1)
        target_logit = 2.2 * objectness + 1.4 * language_similarity
        background_logit = -target_logit + 0.15
        logits = torch.stack([background_logit, target_logit], dim=1)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        uncertainty = entropy_from_logits(logits)

        return {
            "features": features.detach().cpu().numpy().astype(np.float32),
            "logits": logits.detach().cpu().numpy().astype(np.float32),
            "predictions": predictions.detach().cpu().numpy().astype(np.int64),
            "uncertainty": uncertainty.detach().cpu().numpy().astype(np.float32),
            "language_similarity": language_similarity.detach().cpu().numpy().astype(np.float32),
        }
