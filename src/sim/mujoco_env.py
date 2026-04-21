from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from utils.transforms import camera_position_from_transform, camera_transform_to_mujoco_quat


@dataclass
class RenderObservation:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    camera_position: np.ndarray


class MujocoEnv:
    def __init__(self, scene_path: str | Path, image_size: tuple[int, int], config: dict) -> None:
        self.scene_path = str(scene_path)
        self.width, self.height = image_size
        self.config = config

        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        gl_context_cls = getattr(mujoco, "GLContext", None)
        self.gl_context = None
        if gl_context_cls is not None:
            self.gl_context = gl_context_cls(self.width, self.height)
            self.gl_context.make_current()

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.camera_name = config["sim"]["camera_name"]
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self.camera_body_id = self.model.cam_bodyid[self.camera_id]
        self.camera_mocap_id = self.model.body_mocapid[self.camera_body_id]

        if self.camera_mocap_id < 0:
            raise ValueError("scene.xml 의 camera_rig body는 mocap=true 여야 합니다.")

        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_root")
        if self.target_body_id < 0:
            raise ValueError("scene.xml 에 target_root body가 필요합니다.")

        mujoco.mj_forward(self.model, self.data)
        self.current_camera_transform = np.eye(4, dtype=np.float32)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_target_center(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.xpos[self.target_body_id], dtype=np.float32)

    def get_camera_intrinsics(self) -> np.ndarray:
        fovy_rad = np.deg2rad(float(self.model.cam_fovy[self.camera_id]))
        fy = 0.5 * self.height / np.tan(fovy_rad / 2.0)
        fx = fy
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return intrinsics

    def set_camera_pose(self, T_world_camera: np.ndarray) -> None:
        translation = np.asarray(T_world_camera[:3, 3], dtype=np.float32)
        quat_wxyz = camera_transform_to_mujoco_quat(T_world_camera)
        self.data.mocap_pos[self.camera_mocap_id] = translation
        self.data.mocap_quat[self.camera_mocap_id] = quat_wxyz
        mujoco.mj_forward(self.model, self.data)
        self.current_camera_transform = np.asarray(T_world_camera, dtype=np.float32)

    def render(self) -> RenderObservation:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        depth = np.asarray(self.renderer.render(), dtype=np.float32)
        self.renderer.disable_depth_rendering()

        return RenderObservation(
            rgb=rgb,
            depth=depth,
            intrinsics=self.get_camera_intrinsics(),
            T_world_camera=self.current_camera_transform.copy(),
            camera_position=camera_position_from_transform(self.current_camera_transform),
        )
