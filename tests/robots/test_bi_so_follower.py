from unittest.mock import patch

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig


class _FakeSOFollower:
    def __init__(self, config):
        self.config = config
        self.cameras = dict(config.cameras)
        self._side = "left" if config.id and config.id.endswith("_left") else "right"
        self.is_connected = False
        self.is_calibrated = True

    @property
    def _motors_ft(self):
        return {f"joint_{idx}.pos": float for idx in range(7)}

    @property
    def _cameras_ft(self):
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    def connect(self, calibrate=True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def calibrate(self):
        pass

    def configure(self):
        pass

    def setup_motors(self):
        pass

    def get_observation(self):
        obs = {f"joint_{idx}.pos": float(idx) for idx in range(7)}
        for name, cfg in self.config.cameras.items():
            obs[name] = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        return obs

    def send_action(self, action):
        return action


def _make_bi_so_config(cameras=None):
    return BiSOFollowerConfig(
        id="dual",
        left_arm_config=SOFollowerConfig(port="/dev/left"),
        right_arm_config=SOFollowerConfig(port="/dev/right"),
        cameras=cameras or {},
    )


def test_bi_so_top_level_three_cameras_keep_unprefixed_names():
    cameras = {
        "left_wrist": OpenCVCameraConfig(index_or_path=0, width=320, height=240, fps=30),
        "right_wrist": OpenCVCameraConfig(index_or_path=1, width=240, height=320, fps=30),
        "overhead": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }

    with patch("lerobot.robots.bi_so_follower.bi_so_follower.SOFollower", _FakeSOFollower):
        robot = BiSOFollower(_make_bi_so_config(cameras))

    assert robot._cameras_ft == {
        "left_wrist": (240, 320, 3),
        "right_wrist": (320, 240, 3),
        "overhead": (480, 640, 3),
    }
    assert set(robot.cameras) == {"left_wrist", "right_wrist", "overhead"}

    robot.connect()
    obs = robot.get_observation()
    assert "left_wrist" in obs
    assert "right_wrist" in obs
    assert "overhead" in obs
    assert "left_left_wrist" not in obs
    assert "right_right_wrist" not in obs


def test_bi_so_top_level_cameras_and_7dof_arms_make_14dof_features():
    with patch("lerobot.robots.bi_so_follower.bi_so_follower.SOFollower", _FakeSOFollower):
        robot = BiSOFollower(_make_bi_so_config())

    assert len(robot.action_features) == 14
    assert len([name for name in robot.observation_features if name.endswith(".pos")]) == 14
