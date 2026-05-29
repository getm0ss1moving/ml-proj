# Bimanual Training Notes

This checkout supports the planned bimanual dataset layout:

- Two arms with 7 DoF each: `observation.state` and `action` should both be 14-dimensional.
- Three camera views: `observation.images.left_wrist`, `observation.images.right_wrist`, and `observation.images.overhead`.
- Rectangular camera frames can be padded to square frames at dataset-read time.

## Recommended Policy

Use SmolVLA for the planned training and real-robot test. SmolVLA accepts multiple camera features and internally resizes each image with aspect-ratio-preserving padding via `policy.resize_imgs_with_padding`, which defaults to `512x512`. Its default `policy.max_state_dim=32` and `policy.max_action_dim=32` cover the 14-dimensional bimanual state/action vectors, and actions are cropped back to the dataset action dimension before execution.

For the current static-box task, fine-tune from the base checkpoint with conservative small-data settings:

```bash
mkdir -p /data/lry_machine_learning/tmp
export TMPDIR=/data/lry_machine_learning/tmp

CUDA_VISIBLE_DEVICES=0 /data/lry_machine_learning/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.max_state_dim=32 \
  --policy.max_action_dim=32 \
  --policy.resize_imgs_with_padding='(512, 512)' \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=3e-5 \
  --policy.scheduler_warmup_steps=500 \
  --dataset.repo_id=<user>/<bimanual_dataset> \
  --dataset.image_transforms.pad_to_square=true \
  --dataset.image_transforms.square_pad_fill=0 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs.brightness.weight=1.0 \
  --dataset.image_transforms.tfs.brightness.kwargs.brightness='(0.9, 1.1)' \
  --dataset.image_transforms.tfs.contrast.weight=1.0 \
  --dataset.image_transforms.tfs.contrast.kwargs.contrast='(0.9, 1.1)' \
  --dataset.image_transforms.tfs.saturation.weight=0.2 \
  --dataset.image_transforms.tfs.saturation.kwargs.saturation='(0.9, 1.1)' \
  --dataset.image_transforms.tfs.hue.weight=0.0 \
  --dataset.image_transforms.tfs.sharpness.weight=0.5 \
  --dataset.image_transforms.tfs.affine.weight=0.0 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --log_freq=100 \
  --output_dir=/data/lry_machine_learning/lerobot/outputs/train/bimanual_smolvla_49ep_static_box \
  --job_name=bimanual_smolvla_49ep_static_box
```

Run the same command first with `--steps=2000 --save_freq=1000 --batch_size=4` as a shape and loss smoke test. If it is clean, run the full command above and try checkpoints at 5k, 10k, 15k, and 20k on the real robot.

For this 49-episode task, keep geometric augmentation off. The box and two long objects are nearly fixed around the center line, so large affine shifts or rotations teach the model a distribution the robot will not see. Light brightness, contrast, and sharpness changes are useful because camera exposure and shadows vary between training and rollout.

For real-robot testing, use the same `bi_so_follower` camera names and task text used while recording:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/bimanual_smolvla/checkpoints/last/pretrained_model \
  --robot.type=bi_so_follower \
  --robot.cameras="{ left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}" \
  --task="<same task text>" \
  --duration=60
```

If inference latency is too high for smooth motion, switch rollout to RTC:

```bash
lerobot-rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --policy.path=outputs/train/bimanual_smolvla/checkpoints/last/pretrained_model \
  --robot.type=bi_so_follower \
  --robot.cameras="{ left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30}}" \
  --task="<same task text>" \
  --duration=60
```

ACT remains useful as a lightweight baseline. ACT processes each image feature independently, so it can consume three cameras with different spatial resolutions after square padding. Policies that stack cameras into one tensor, such as Diffusion, VQ-BeT, and MultiTaskDiT, still require matching image shapes across cameras unless an additional resize step is added.

## Square Padding

Enable square padding from the training CLI:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<user>/<dataset> \
  --dataset.image_transforms.pad_to_square=true
```

This pads only the shorter side with black pixels and does not resize the image. For example, `240x320` becomes `320x320`, while `480x640` becomes `640x640`.

Random image augmentations remain controlled by `--dataset.image_transforms.enable=true`. Square padding can be enabled by itself. For SmolVLA this dataset transform is optional because the policy already resizes with padding internally; enable it only if you want stored rectangular frames squared before policy preprocessing.

## Robot Configuration

For `bi_so_follower`, shared top-level cameras can be configured under `--robot.cameras`. Use unique camera names such as `left_wrist`, `right_wrist`, and `overhead`; these names remain unprefixed in dataset features. Joint features remain prefixed as `left_*` and `right_*`, producing 14-dimensional state/action tensors when each arm exposes 7 position joints.
