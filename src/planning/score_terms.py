from __future__ import annotations

import math
import numpy as np


def _safe_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-6, None)
    return vectors / norms


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / max(np.linalg.norm(a), 1e-6)
    b_n = b / max(np.linalg.norm(b), 1e-6)
    cosine = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def visibility_weights(candidate_position: np.ndarray, gaussian_positions: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    if len(gaussian_positions) == 0:
        return np.empty((0,), dtype=np.float32)

    outward_normals = _safe_normalize(gaussian_positions - target_center.reshape(1, 3))
    to_camera = _safe_normalize(candidate_position.reshape(1, 3) - gaussian_positions)
    weights = np.clip(np.sum(outward_normals * to_camera, axis=1), 0.0, 1.0)
    return weights.astype(np.float32)


def move_cost(candidate_position: np.ndarray, current_position: np.ndarray, target_center: np.ndarray) -> float:
    distance_cost = float(np.linalg.norm(candidate_position - current_position))
    current_vec = current_position - target_center
    candidate_vec = candidate_position - target_center
    angular_cost = angular_distance_deg(current_vec, candidate_vec) / 180.0
    return distance_cost + angular_cost


def novelty_bonus(candidate_position: np.ndarray, target_center: np.ndarray, past_positions: list[np.ndarray]) -> float:
    if not past_positions:
        return 1.0

    candidate_vec = candidate_position - target_center
    distances = [
        angular_distance_deg(candidate_vec, np.asarray(past_position, dtype=np.float32) - target_center)
        for past_position in past_positions
    ]
    min_angle = min(distances)
    return float(np.clip(min_angle / 90.0, 0.0, 1.0))


def score_candidate(
    candidate: dict,
    current_position: np.ndarray,
    target_center: np.ndarray,
    state_arrays: dict[str, np.ndarray],
    past_positions: list[np.ndarray],
    weights: dict,
    revisit_penalty: float,
) -> dict:
    candidate_position = candidate["T_world_camera"][:3, 3]
    positions = state_arrays["positions"]

    if len(positions) == 0:
        move = move_cost(candidate_position, current_position, target_center)
        score = 0.5 - weights["move_cost"] * move
        return {
            **candidate,
            "score": float(score),
            "delta_u": 0.0,
            "lang_affinity": 0.0,
            "consistency_gain": 0.0,
            "occlusion_relief": 0.0,
            "move_cost": float(move),
        }

    visibility = visibility_weights(candidate_position, positions, target_center)
    target_weight = np.clip((state_arrays["language_similarity"] + 1.0) * 0.5, 0.0, 1.0)
    novelty = novelty_bonus(candidate_position, target_center, past_positions)

    delta_u = float(np.mean(state_arrays["uncertainty"] * target_weight * visibility) * novelty)
    lang_affinity = float(np.mean(target_weight * visibility))
    consistency_gain = float(np.mean((1.0 - state_arrays["reliability"]) * target_weight * visibility) * novelty)
    occlusion_relief = float(
        np.mean((1.0 / np.maximum(state_arrays["view_count"], 1.0)) * target_weight * visibility) * novelty
    )
    move = move_cost(candidate_position, current_position, target_center)
    score = (
        weights["delta_u"] * delta_u
        + weights["lang_affinity"] * lang_affinity
        + weights["consistency_gain"] * consistency_gain
        + weights["occlusion_relief"] * occlusion_relief
        - weights["move_cost"] * move
        - revisit_penalty * (1.0 - novelty)
    )

    return {
        **candidate,
        "score": float(score),
        "delta_u": delta_u,
        "lang_affinity": lang_affinity,
        "consistency_gain": consistency_gain,
        "occlusion_relief": occlusion_relief,
        "move_cost": float(move),
    }
