# Bimanual Training Notes

This checkout supports the planned bimanual dataset layout:

- Two arms with 7 DoF each: `observation.state` and `action` should both be 14-dimensional.
- Three camera views: `observation.images.left_wrist`, `observation.images.right_wrist`, and `observation.images.overhead`.
- Rectangular camera frames can be padded to square frames at dataset-read time.

## Recommended Policy

Use ACT for the first training runs. ACT processes each image feature independently, so it can consume three cameras with different spatial resolutions after square padding. Policies that stack cameras into one tensor, such as Diffusion, VQ-BeT, and MultiTaskDiT, still require matching image shapes across cameras unless an additional resize step is added.

## Square Padding

Enable square padding from the training CLI:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=<user>/<dataset> \
  --dataset.image_transforms.pad_to_square=true
```

This pads only the shorter side with black pixels and does not resize the image. For example, `240x320` becomes `320x320`, while `480x640` becomes `640x640`.

Random image augmentations remain controlled by `--dataset.image_transforms.enable=true`. Square padding can be enabled by itself.

## Robot Configuration

For `bi_so_follower`, shared top-level cameras can be configured under `--robot.cameras`. Use unique camera names such as `left_wrist`, `right_wrist`, and `overhead`; these names remain unprefixed in dataset features. Joint features remain prefixed as `left_*` and `right_*`, producing 14-dimensional state/action tensors when each arm exposes 7 position joints.
