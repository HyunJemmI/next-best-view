from __future__ import annotations


class ReachabilityEvaluator:
    """첫 MVP에서는 arm feasibility를 stub으로 둔다."""

    def evaluate(self, candidate: dict) -> dict:
        return {
            "reachable": True,
            "ik_cost": 0.0,
            "collision_free": True,
        }
