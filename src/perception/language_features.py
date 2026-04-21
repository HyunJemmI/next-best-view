from __future__ import annotations

import hashlib

import torch


class ToyLanguageEncoder:
    def __init__(self, embedding_dim: int = 32) -> None:
        self.embedding_dim = embedding_dim

    def _seed_from_text(self, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def encode(self, text: str, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed_from_text(text))
        embedding = torch.randn(self.embedding_dim, generator=generator, dtype=torch.float32)
        embedding = embedding / embedding.norm().clamp(min=1e-6)
        return embedding.to(device)
