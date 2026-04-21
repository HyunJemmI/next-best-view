from __future__ import annotations

import numpy as np

from planning.reachability import ReachabilityEvaluator
from planning.score_terms import angular_distance_deg, score_candidate
from planning.view_sampler import OrbitViewSampler


class NBVPlanner:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.sampler = OrbitViewSampler(config)
        self.reachability = ReachabilityEvaluator()
        self.weights = config["planning"]["weights"]
        self.revisit_penalty = float(config["planning"]["revisit_penalty"])
        self.min_view_novelty_deg = float(config["planning"]["min_view_novelty_deg"])

    def sample_candidates(self, target_center: np.ndarray) -> list[dict]:
        return self.sampler.sample(target_center)

    def score_candidates(
        self,
        candidates: list[dict],
        current_pose: np.ndarray,
        target_center: np.ndarray,
        state_arrays: dict[str, np.ndarray],
        past_positions: list[np.ndarray],
    ) -> list[dict]:
        current_position = current_pose[:3, 3]
        scored = []
        for candidate in candidates:
            candidate_position = candidate["T_world_camera"][:3, 3]
            if past_positions:
                candidate_vector = candidate_position - target_center
                min_angle = min(
                    angular_distance_deg(
                        candidate_vector,
                        np.asarray(past_position, dtype=np.float32) - target_center,
                    )
                    for past_position in past_positions
                )
                if min_angle < self.min_view_novelty_deg:
                    continue

            reachability = self.reachability.evaluate(candidate)
            if not reachability["reachable"]:
                continue
            candidate_score = score_candidate(
                candidate=candidate,
                current_position=current_position,
                target_center=target_center,
                state_arrays=state_arrays,
                past_positions=past_positions,
                weights=self.weights,
                revisit_penalty=self.revisit_penalty,
            )
            candidate_score.update(reachability)
            scored.append(candidate_score)

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def select_next_view(self, scored_candidates: list[dict]) -> dict:
        if not scored_candidates:
            raise ValueError("유효한 NBV 후보가 없습니다.")
        return scored_candidates[0]
