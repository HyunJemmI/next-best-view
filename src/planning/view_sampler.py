from __future__ import annotations

import numpy as np

from sim.camera_utils import candidate_orbit_transforms


class OrbitViewSampler:
    def __init__(self, config: dict) -> None:
        self.radius = float(config["sim"]["candidate_radius"])
        self.azimuth_samples = int(config["sim"]["azimuth_samples"])
        self.elevation_candidates_deg = [float(value) for value in config["sim"]["elevation_candidates_deg"]]
        self.world_up = np.asarray(config["sim"]["world_up"], dtype=np.float32)

    def sample(self, target_center: np.ndarray) -> list[dict]:
        return candidate_orbit_transforms(
            target_center=target_center,
            radius=self.radius,
            azimuth_samples=self.azimuth_samples,
            elevation_candidates_deg=self.elevation_candidates_deg,
            world_up=self.world_up,
        )
