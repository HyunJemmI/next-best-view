from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from utils.transforms import interpolate_transforms


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _open3d():
    import open3d as o3d

    return o3d


def depth_to_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        values = depth[valid]
        lo, hi = np.percentile(values, [2, 98])
        preview[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return preview


def _new_agg_figure(figsize: tuple[float, float]):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def save_depth_preview(path: str | Path, depth: np.ndarray) -> None:
    preview = depth_to_preview(depth)
    figure = _new_agg_figure((6, 4))
    axis = figure.add_subplot(111)
    axis.imshow(preview, cmap="viridis")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", pad_inches=0)


def save_candidate_plot(path: str | Path, scored_candidates: list[dict], selected_index: int | None = None) -> None:
    if not scored_candidates:
        return

    azimuths = [candidate["azimuth_deg"] for candidate in scored_candidates]
    elevations = [candidate["elevation_deg"] for candidate in scored_candidates]
    scores = [candidate["score"] for candidate in scored_candidates]

    figure = _new_agg_figure((7, 5))
    axis = figure.add_subplot(111)
    scatter = axis.scatter(azimuths, elevations, c=scores, cmap="plasma", s=90)
    if selected_index is not None and 0 <= selected_index < len(scored_candidates):
        axis.scatter(
            [azimuths[selected_index]],
            [elevations[selected_index]],
            s=220,
            facecolors="none",
            edgecolors="white",
            linewidths=2.5,
        )
    axis.set_xlabel("Azimuth (deg)")
    axis.set_ylabel("Elevation (deg)")
    axis.set_title("NBV Candidate Scores")
    figure.colorbar(scatter, ax=axis, label="score")
    figure.tight_layout()
    figure.savefig(path)


def save_map_snapshot(path: str | Path, positions: np.ndarray, colors: np.ndarray, target_center: np.ndarray) -> None:
    figure = _new_agg_figure((6, 6))
    axis = figure.add_subplot(111)
    if len(positions):
        axis.scatter(positions[:, 0], positions[:, 1], c=colors, s=8)
    axis.scatter([target_center[0]], [target_center[1]], c="red", s=80, marker="x")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Top-down Global Map Snapshot")
    axis.axis("equal")
    figure.tight_layout()
    figure.savefig(path)


def save_pointcloud(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors_uint8, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


class MujocoSceneViewer:
    def __init__(self, env, target_center: np.ndarray) -> None:
        self.env = env
        self.target_center = np.asarray(target_center, dtype=np.float32)
        self.available = False
        self.viewer = None
        self.error: str | None = None

        try:
            import mujoco
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                env.model,
                env.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = self.target_center
                self.viewer.cam.distance = 1.35
                self.viewer.cam.azimuth = 132.0
                self.viewer.cam.elevation = -24.0
            self.viewer.sync()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def sync(self) -> None:
        if not self.available or self.viewer is None:
            return
        if not self.viewer.is_running():
            self.available = False
            return
        self.viewer.sync()

    def animate_camera_transition(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        transition_steps: int,
        fps: float,
    ) -> None:
        if not self.available:
            self.env.set_camera_pose(end_pose)
            return

        frame_sleep = 1.0 / max(fps, 1e-3)
        for transform in interpolate_transforms(start_pose, end_pose, transition_steps):
            self.env.set_camera_pose(transform)
            self.sync()
            time.sleep(frame_sleep)

    def hold(self, seconds: float) -> None:
        if not self.available:
            return
        end_time = time.time() + max(seconds, 0.0)
        while time.time() < end_time and self.available:
            self.sync()
            time.sleep(0.02)

    def close(self) -> None:
        if self.viewer is not None and hasattr(self.viewer, "close"):
            self.viewer.close()


class Open3DMapViewer:
    def __init__(self, target_center: np.ndarray, point_size: float = 4.0) -> None:
        o3d = _open3d()
        self.target_center = np.asarray(target_center, dtype=np.float32)
        self.point_size = float(point_size)
        self.available = False
        self.error: str | None = None
        self.o3d = o3d
        self.vis = None
        self.point_cloud = o3d.geometry.PointCloud()
        self.camera_path = o3d.geometry.LineSet()
        self.target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        self.target_frame.translate(self.target_center.astype(np.float64))

        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="NBV Global Map",
                width=960,
                height=720,
                visible=True,
            )
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = np.array([0.08, 0.09, 0.11], dtype=np.float64)
            self.vis.add_geometry(self.point_cloud)
            self.vis.add_geometry(self.camera_path)
            self.vis.add_geometry(self.target_frame)
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def update(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        camera_positions: list[np.ndarray],
    ) -> None:
        if not self.available or self.vis is None:
            return

        o3d = self.o3d

        if len(positions):
            self.point_cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        else:
            self.point_cloud.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            self.point_cloud.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

        if len(camera_positions) >= 2:
            path_points = np.asarray(camera_positions, dtype=np.float64)
            path_lines = np.array([[idx, idx + 1] for idx in range(len(path_points) - 1)], dtype=np.int32)
            path_colors = np.tile(np.array([[1.0, 0.58, 0.1]], dtype=np.float64), (len(path_lines), 1))
            self.camera_path.points = o3d.utility.Vector3dVector(path_points)
            self.camera_path.lines = o3d.utility.Vector2iVector(path_lines)
            self.camera_path.colors = o3d.utility.Vector3dVector(path_colors)
        elif len(camera_positions) == 1:
            path_points = np.asarray(camera_positions, dtype=np.float64)
            self.camera_path.points = o3d.utility.Vector3dVector(path_points)
            self.camera_path.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
            self.camera_path.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))

        self.vis.update_geometry(self.point_cloud)
        self.vis.update_geometry(self.camera_path)
        self.vis.poll_events()
        self.vis.update_renderer()

    def hold(self, seconds: float) -> None:
        if not self.available or self.vis is None:
            return
        end_time = time.time() + max(seconds, 0.0)
        while time.time() < end_time:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.02)

    def close(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()


class LiveDebugViewer:
    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self.plt = None
        self.fig = None
        self.axes = None
        self.rgb_artist = None
        self.depth_artist = None
        self.score_colorbar = None

        try:
            plt = _pyplot()
            plt.ion()
            self.plt = plt
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.axes = np.asarray(self.axes)
            self.axes[0, 0].set_title("Active RGB")
            self.axes[0, 1].set_title("Active Depth")
            self.axes[1, 0].set_title("Top-down Global Map")
            self.axes[1, 1].set_title("Candidate Scores")
            self.rgb_artist = self.axes[0, 0].imshow(np.zeros((64, 64, 3), dtype=np.float32))
            self.depth_artist = self.axes[0, 1].imshow(np.zeros((64, 64), dtype=np.float32), cmap="viridis")
            for axis in self.axes.reshape(-1):
                axis.axis("off")
            self.axes[1, 0].axis("on")
            self.axes[1, 1].axis("on")
            self.fig.tight_layout()
            self.fig.show()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        map_positions: np.ndarray,
        map_colors: np.ndarray,
        target_center: np.ndarray,
        scored_candidates: list[dict] | None,
    ) -> None:
        if not self.available or self.plt is None or self.axes is None:
            return

        self.rgb_artist.set_data(rgb)
        self.depth_artist.set_data(depth_to_preview(depth))

        map_axis = self.axes[1, 0]
        map_axis.clear()
        map_axis.set_title("Top-down Global Map")
        if len(map_positions):
            map_axis.scatter(map_positions[:, 0], map_positions[:, 1], c=map_colors, s=8)
        map_axis.scatter([target_center[0]], [target_center[1]], c="red", s=90, marker="x")
        map_axis.axis("equal")
        map_axis.grid(alpha=0.2)

        score_axis = self.axes[1, 1]
        if self.score_colorbar is not None:
            self.score_colorbar.remove()
            self.score_colorbar = None
        score_axis.clear()
        score_axis.set_title("Candidate Scores")
        if scored_candidates:
            azimuths = [candidate["azimuth_deg"] for candidate in scored_candidates]
            elevations = [candidate["elevation_deg"] for candidate in scored_candidates]
            scores = [candidate["score"] for candidate in scored_candidates]
            scatter = score_axis.scatter(azimuths, elevations, c=scores, cmap="plasma", s=85)
            score_axis.scatter(
                [azimuths[0]],
                [elevations[0]],
                s=220,
                facecolors="none",
                edgecolors="white",
                linewidths=2.2,
            )
            self.score_colorbar = self.fig.colorbar(scatter, ax=score_axis, fraction=0.046, pad=0.04)
        score_axis.set_xlabel("Azimuth (deg)")
        score_axis.set_ylabel("Elevation (deg)")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def hold(self, seconds: float) -> None:
        if not self.available or self.plt is None:
            return
        end_time = time.time() + max(seconds, 0.0)
        while time.time() < end_time:
            self.plt.pause(0.02)

    def close(self) -> None:
        if self.available and self.plt is not None and self.fig is not None:
            self.plt.close(self.fig)
