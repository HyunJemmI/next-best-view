from __future__ import annotations

import numpy as np
import open3d as o3d


def rgbd_to_world_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
    depth_min: float,
    depth_max: float,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth[grid_y, grid_x]
    valid = np.isfinite(sampled_depth) & (sampled_depth > depth_min) & (sampled_depth < depth_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    z = sampled_depth[valid].astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    camera_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world_points_h = (T_world_camera @ camera_points.T).T
    world_points = world_points_h[:, :3].astype(np.float32)
    colors = rgb[grid_y[valid], grid_x[valid]].astype(np.float32) / 255.0
    return world_points, colors


def make_open3d_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud
