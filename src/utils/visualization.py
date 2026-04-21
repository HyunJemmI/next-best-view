from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def save_depth_preview(path: str | Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        values = depth[valid]
        lo, hi = np.percentile(values, [2, 98])
        preview[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    plt.figure(figsize=(6, 4))
    plt.imshow(preview, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_candidate_plot(path: str | Path, scored_candidates: list[dict], selected_index: int | None = None) -> None:
    if not scored_candidates:
        return

    azimuths = [candidate["azimuth_deg"] for candidate in scored_candidates]
    elevations = [candidate["elevation_deg"] for candidate in scored_candidates]
    scores = [candidate["score"] for candidate in scored_candidates]

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(azimuths, elevations, c=scores, cmap="plasma", s=90)
    if selected_index is not None and 0 <= selected_index < len(scored_candidates):
        plt.scatter(
            [azimuths[selected_index]],
            [elevations[selected_index]],
            s=220,
            facecolors="none",
            edgecolors="white",
            linewidths=2.5,
        )
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("NBV Candidate Scores")
    plt.colorbar(scatter, label="score")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_map_snapshot(path: str | Path, positions: np.ndarray, colors: np.ndarray, target_center: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    if len(positions):
        plt.scatter(positions[:, 0], positions[:, 1], c=colors, s=8)
    plt.scatter([target_center[0]], [target_center[1]], c="red", s=80, marker="x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Top-down Global Map Snapshot")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_pointcloud(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)
