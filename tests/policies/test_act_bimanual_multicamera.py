import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def test_act_forward_with_three_rectangular_cameras_and_14dof_state_action():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 48)),
            "observation.images.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 32)),
            "observation.images.overhead": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(config)
    policy.train()

    batch = {
        OBS_STATE: torch.randn(1, 14),
        "observation.images.left_wrist": torch.randn(1, 3, 32, 48),
        "observation.images.right_wrist": torch.randn(1, 3, 48, 32),
        "observation.images.overhead": torch.randn(1, 3, 64, 64),
        ACTION: torch.randn(1, 2, 14),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }

    loss, output_dict = policy(batch)

    assert loss.isfinite()
    assert "l1_loss" in output_dict
