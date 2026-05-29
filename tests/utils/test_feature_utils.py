from lerobot.configs import FeatureType
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features, hw_to_dataset_features


def test_bimanual_14dof_features_with_three_rectangular_cameras():
    joint_features = {
        **{f"left_joint_{idx}.pos": float for idx in range(7)},
        **{f"right_joint_{idx}.pos": float for idx in range(7)},
    }
    camera_features = {
        "left_wrist": (240, 320, 3),
        "right_wrist": (320, 240, 3),
        "overhead": (480, 640, 3),
    }

    obs_features = hw_to_dataset_features({**joint_features, **camera_features}, OBS_STR)
    action_features = hw_to_dataset_features(joint_features, ACTION)
    policy_features = dataset_to_policy_features({**obs_features, **action_features})

    assert obs_features[OBS_STATE]["shape"] == (14,)
    assert obs_features[OBS_STATE]["names"] == list(joint_features)
    assert action_features[ACTION]["shape"] == (14,)
    assert action_features[ACTION]["names"] == list(joint_features)

    assert policy_features[OBS_STATE].type is FeatureType.STATE
    assert policy_features[OBS_STATE].shape == (14,)
    assert policy_features[ACTION].type is FeatureType.ACTION
    assert policy_features[ACTION].shape == (14,)

    assert policy_features["observation.images.left_wrist"].shape == (3, 240, 320)
    assert policy_features["observation.images.right_wrist"].shape == (3, 320, 240)
    assert policy_features["observation.images.overhead"].shape == (3, 480, 640)
