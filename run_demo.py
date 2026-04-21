from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.gaussian_semantic_wrapper import GaussianSemanticWrapper
from perception.fusion import GlobalMap
from perception.pointcloud import rgbd_to_world_points
from planning.nbv import NBVPlanner
from sim.camera_utils import orbit_camera_transform
from sim.mujoco_env import MujocoEnv
from utils.io import ensure_output_dirs, load_config, save_json, save_rgb_image
from utils.visualization import save_candidate_plot, save_depth_preview, save_map_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBV macOS minimal MVP demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment_report() -> dict:
    package_versions = {}
    for package_name in ["mujoco", "open3d", "torch", "numpy", "yaml", "matplotlib"]:
        module = importlib.import_module(package_name if package_name != "yaml" else "yaml")
        version = getattr(module, "__version__", "unknown")
        package_versions[package_name] = version

    return {
        "python": sys.version,
        "packages": package_versions,
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def run_env_check(config: dict) -> None:
    print(json.dumps(environment_report(), indent=2, ensure_ascii=False))
    env = MujocoEnv(
        scene_path=PROJECT_ROOT / "assets" / "mujoco" / "scene.xml",
        image_size=(config["sim"]["image_width"], config["sim"]["image_height"]),
        config=config,
    )
    target_center = env.get_target_center()
    initial_pose = orbit_camera_transform(
        target_center=target_center,
        radius=float(config["sim"]["initial_radius"]),
        azimuth_deg=float(config["sim"]["initial_azimuth_deg"]),
        elevation_deg=float(config["sim"]["initial_elevation_deg"]),
        world_up=np.asarray(config["sim"]["world_up"], dtype=np.float32),
    )
    env.set_camera_pose(initial_pose)
    observation = env.render()
    points, colors = rgbd_to_world_points(
        rgb=observation.rgb,
        depth=observation.depth,
        intrinsics=observation.intrinsics,
        T_world_camera=observation.T_world_camera,
        depth_min=float(config["perception"]["depth_min"]),
        depth_max=float(config["perception"]["depth_max"]),
        stride=int(config["perception"]["point_stride"]),
    )
    report = {
        "target_center": target_center.tolist(),
        "rgb_shape": list(observation.rgb.shape),
        "depth_shape": list(observation.depth.shape),
        "depth_min": float(np.min(observation.depth)),
        "depth_max": float(np.max(observation.depth)),
        "point_count": int(len(points)),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))

    if args.check_env:
        run_env_check(config)
        return

    output_paths = ensure_output_dirs(PROJECT_ROOT / config["outputs"]["base_dir"])

    env = MujocoEnv(
        scene_path=PROJECT_ROOT / "assets" / "mujoco" / "scene.xml",
        image_size=(config["sim"]["image_width"], config["sim"]["image_height"]),
        config=config,
    )
    target_center = env.get_target_center()
    model = GaussianSemanticWrapper(config)
    global_map = GlobalMap(voxel_size=float(config["perception"]["gaussian_voxel_size"]))
    planner = NBVPlanner(config)

    iterations = int(args.iterations or config["planning"]["iterations"])
    current_pose = orbit_camera_transform(
        target_center=target_center,
        radius=float(config["sim"]["initial_radius"]),
        azimuth_deg=float(config["sim"]["initial_azimuth_deg"]),
        elevation_deg=float(config["sim"]["initial_elevation_deg"]),
        world_up=np.asarray(config["sim"]["world_up"], dtype=np.float32),
    )

    summaries = []
    for iteration in range(iterations):
        env.set_camera_pose(current_pose)
        rendered = env.render()

        save_rgb_image(output_paths["debug"] / f"iter_{iteration:02d}_rgb.png", rendered.rgb)
        save_depth_preview(output_paths["debug"] / f"iter_{iteration:02d}_depth.png", rendered.depth)

        points, colors = rgbd_to_world_points(
            rgb=rendered.rgb,
            depth=rendered.depth,
            intrinsics=rendered.intrinsics,
            T_world_camera=rendered.T_world_camera,
            depth_min=float(config["perception"]["depth_min"]),
            depth_max=float(config["perception"]["depth_max"]),
            stride=int(config["perception"]["point_stride"]),
        )

        observation = {
            "points": points,
            "colors": colors,
            "target_center": target_center,
            "query_text": config["query"]["text"],
        }
        model_output = model.infer(observation)
        global_map.update(
            observation=observation,
            model_output=model_output,
            view_id=iteration,
            camera_position=rendered.camera_position,
        )

        state_arrays = global_map.get_state_arrays()
        save_map_snapshot(
            output_paths["debug"] / f"iter_{iteration:02d}_map_topdown.png",
            positions=state_arrays["positions"],
            colors=state_arrays["colors"],
            target_center=target_center,
        )
        global_map.save_pointcloud(output_paths["pcd"] / f"iter_{iteration:02d}_global_map.ply")

        iteration_summary = {
            "iteration": iteration,
            "camera_position": rendered.camera_position.tolist(),
            "observation_points": int(len(points)),
            "gaussian_proxy_count": int(len(state_arrays["positions"])),
            "mean_uncertainty": float(np.mean(state_arrays["uncertainty"])) if len(state_arrays["uncertainty"]) else 0.0,
            "mean_language_similarity": float(np.mean(state_arrays["language_similarity"])) if len(state_arrays["language_similarity"]) else 0.0,
            "device_report": model.device_report.__dict__,
        }

        if iteration < iterations - 1:
            candidates = planner.sample_candidates(target_center)
            scored = planner.score_candidates(
                candidates=candidates,
                current_pose=current_pose,
                target_center=target_center,
                state_arrays=state_arrays,
                past_positions=global_map.view_positions,
            )
            selected = planner.select_next_view(scored)
            current_pose = selected["T_world_camera"]

            iteration_summary["selected_view"] = {
                "azimuth_deg": selected["azimuth_deg"],
                "elevation_deg": selected["elevation_deg"],
                "score": selected["score"],
                "delta_u": selected["delta_u"],
                "lang_affinity": selected["lang_affinity"],
                "consistency_gain": selected["consistency_gain"],
                "occlusion_relief": selected["occlusion_relief"],
                "move_cost": selected["move_cost"],
            }
            save_candidate_plot(
                output_paths["debug"] / f"iter_{iteration:02d}_candidate_scores.png",
                scored_candidates=scored,
                selected_index=0,
            )

        summaries.append(iteration_summary)
        print(json.dumps(iteration_summary, ensure_ascii=False))

    save_json(output_paths["logs"] / "run_summary.json", summaries)


if __name__ == "__main__":
    main()
