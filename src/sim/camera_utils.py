from __future__ import annotations

import math
import numpy as np

from utils.transforms import look_at_camera_transform


def orbit_position(
    target_center: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)
    return np.asarray(target_center, dtype=np.float32) + np.array([x, y, z], dtype=np.float32)


def orbit_camera_transform(
    target_center: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    position = orbit_position(target_center, radius, azimuth_deg, elevation_deg)
    return look_at_camera_transform(position, target_center, world_up=world_up)


def candidate_orbit_transforms(
    target_center: np.ndarray,
    radius: float,
    azimuth_samples: int,
    elevation_candidates_deg: list[float],
    world_up: np.ndarray | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    for elevation_deg in elevation_candidates_deg:
        for azimuth_deg in np.linspace(-180.0, 180.0, num=azimuth_samples, endpoint=False):
            transform = orbit_camera_transform(
                target_center=target_center,
                radius=radius,
                azimuth_deg=float(azimuth_deg),
                elevation_deg=float(elevation_deg),
                world_up=world_up,
            )
            candidates.append(
                {
                    "azimuth_deg": float(azimuth_deg),
                    "elevation_deg": float(elevation_deg),
                    "T_world_camera": transform,
                }
            )
    return candidates
