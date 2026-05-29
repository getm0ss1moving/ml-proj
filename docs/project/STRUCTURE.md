# Repository Structure

This repository keeps the standard Python `src` layout while preserving LeRobot's upstream module boundaries.

- `src/lerobot/`: core package code. Do not move policy, dataset, robot, environment, processor, or script modules as part of housekeeping.
- `tests/`: pytest suite and test fixtures.
- `docs/source/`: upstream LeRobot documentation sources.
- `docs/project/`: local structure and maintenance notes for this optimized checkout.
- `examples/`: runnable examples and tutorials.
- `benchmarks/`: benchmarking utilities.
- `scripts/`: repository and CI helper scripts.
- `docker/`: container build files.
- `media/`: README and documentation media assets.

Root-level files are limited to packaging, project metadata, contributor guidance, agent guidance, and repository policy files. Runtime outputs, caches, logs, process markers, and ad-hoc launch files should stay untracked.
