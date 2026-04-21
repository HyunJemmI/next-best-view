from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


BODY_TO_CV_ROTATION = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def look_at_body_rotation(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target_position = np.asarray(target_position, dtype=np.float32)
    world_up = np.asarray(world_up if world_up is not None else [0.0, 0.0, 1.0], dtype=np.float32)

    forward = normalize(target_position - camera_position)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-5:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)

    right = normalize(right)
    up = normalize(np.cross(right, forward))
    back = normalize(-forward)
    rotation = np.column_stack([right, up, back]).astype(np.float32)
    return rotation


def look_at_camera_transform(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    body_rotation = look_at_body_rotation(camera_position, target_position, world_up)
    cv_rotation = body_rotation @ BODY_TO_CV_ROTATION
    return make_transform(cv_rotation, np.asarray(camera_position, dtype=np.float32))


def camera_transform_to_mujoco_quat(transform_world_camera_cv: np.ndarray) -> np.ndarray:
    body_rotation = transform_world_camera_cv[:3, :3] @ BODY_TO_CV_ROTATION
    quat_xyzw = Rotation.from_matrix(body_rotation).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    return quat_wxyz


def camera_position_from_transform(transform_world_camera_cv: np.ndarray) -> np.ndarray:
    return np.asarray(transform_world_camera_cv[:3, 3], dtype=np.float32)


def interpolate_transforms(
    start_transform: np.ndarray,
    end_transform: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    if num_steps <= 1:
        return [np.asarray(end_transform, dtype=np.float32)]

    key_times = [0.0, 1.0]
    rotations = Rotation.from_matrix(
        np.stack([start_transform[:3, :3], end_transform[:3, :3]], axis=0)
    )
    slerp = Slerp(key_times, rotations)
    translations = np.stack([start_transform[:3, 3], end_transform[:3, 3]], axis=0)

    transforms: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, num=num_steps, endpoint=True):
        rotation = slerp([float(alpha)]).as_matrix()[0]
        translation = (1.0 - alpha) * translations[0] + alpha * translations[1]
        transforms.append(make_transform(rotation, translation))
    return transforms
