import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


pytest.importorskip("transformers")

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad  # noqa: E402


def _make_bimanual_config() -> SmolVLAConfig:
    return SmolVLAConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            f"{OBS_IMAGES}.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
            f"{OBS_IMAGES}.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            f"{OBS_IMAGES}.overhead": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        resize_imgs_with_padding=(64, 64),
        max_state_dim=32,
        max_action_dim=32,
    )


def test_smolvla_resizes_rectangular_cameras_with_padding():
    wide = torch.ones(1, 3, 24, 32)
    tall = torch.ones(1, 3, 32, 24)

    wide_padded = resize_with_pad(wide, width=64, height=64, pad_value=0)
    tall_padded = resize_with_pad(tall, width=64, height=64, pad_value=0)

    assert wide_padded.shape == (1, 3, 64, 64)
    assert tall_padded.shape == (1, 3, 64, 64)
    assert torch.all(wide_padded[:, :, :16, :] == 0)
    assert torch.all(tall_padded[:, :, :, :16] == 0)
    assert torch.all(wide_padded[:, :, 16:, :] == 1)
    assert torch.all(tall_padded[:, :, :, 16:] == 1)


def test_smolvla_prepares_three_cameras_and_14d_bimanual_tensors():
    config = _make_bimanual_config()
    policy = object.__new__(SmolVLAPolicy)
    policy.config = config

    batch = {
        OBS_STATE: torch.arange(14, dtype=torch.float32).reshape(1, 14),
        ACTION: torch.ones(1, config.chunk_size, 14),
        f"{OBS_IMAGES}.left_wrist": torch.zeros(1, 3, 240, 320),
        f"{OBS_IMAGES}.right_wrist": torch.zeros(1, 3, 480, 640),
        f"{OBS_IMAGES}.overhead": torch.zeros(1, 3, 720, 1280),
    }

    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)

    assert len(images) == 3
    assert len(masks) == 3
    assert all(image.shape == (1, 3, 64, 64) for image in images)
    assert all(mask.tolist() == [True] for mask in masks)
    assert state.shape == (1, 32)
    assert actions.shape == (1, config.chunk_size, 32)
    assert torch.equal(state[:, :14], batch[OBS_STATE])
    assert torch.all(state[:, 14:] == 0)
    assert torch.all(actions[:, :, :14] == 1)
    assert torch.all(actions[:, :, 14:] == 0)


def test_smolvla_config_can_store_dataset_action_names_for_rollout_order():
    config = _make_bimanual_config()
    config.action_feature_names = [f"joint_{idx}.pos" for idx in range(14)]

    assert config.action_feature_names == [f"joint_{idx}.pos" for idx in range(14)]
