# Maintenance Notes

## Disk-Space Policy

The working copy lives at `/data/lry_machine_learning/lerobot`. The host root filesystem has limited free space, so commands that create temporary files should prefer `/data/lry_machine_learning/tmp` when possible:

```bash
TMPDIR=/data/lry_machine_learning/tmp <command>
```

## Cleanup Policy

The following are considered disposable local artifacts and should not be committed:

- `outputs/`, `logs/`, `runs/`, `checkpoints/`
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`
- `train_*.log`, `*.logx`, `*.pid`, swap files, and ad-hoc launch files

Use a dry run before deleting ignored files:

```bash
git clean -ndX
```

Then clean ignored runtime artifacts only when the output is understood:

```bash
git clean -fdX
```

Untracked files should be reviewed separately with `git clean -nd` before deletion.

## 2026-05-29 Cleanup

Removed ignored runtime outputs and caches from the remote checkout, including `outputs/`, `logs/`, Python bytecode caches, `src/lerobot.egg-info/`, swap files, and local training logs. This reduced the checkout from roughly 22 GB to roughly 394 MB without changing core source files.
