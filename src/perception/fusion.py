from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.gaussian_state import GaussianState
from utils.visualization import save_pointcloud


class GlobalMap:
    def __init__(self, voxel_size: float) -> None:
        self.state = GaussianState(voxel_size=voxel_size)
        self.view_positions: list[np.ndarray] = []

    def update(self, observation: dict, model_output: dict, view_id: int, camera_position: np.ndarray) -> None:
        self.state.update(
            points=observation["points"],
            colors=observation["colors"],
            features=model_output["features"],
            logits=model_output["logits"],
            uncertainty=model_output["uncertainty"],
            language_similarity=model_output["language_similarity"],
            view_id=view_id,
        )
        self.view_positions.append(np.asarray(camera_position, dtype=np.float32))

    def get_state_arrays(self) -> dict[str, np.ndarray]:
        return self.state.to_arrays()

    def save_pointcloud(self, path: str | Path) -> None:
        arrays = self.get_state_arrays()
        save_pointcloud(path, arrays["positions"], arrays["colors"])
