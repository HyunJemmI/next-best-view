from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSemanticModel(ABC):
    @abstractmethod
    def infer(self, observation: dict) -> dict:
        raise NotImplementedError
