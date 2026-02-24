# Repository Guidelines

## Project Structure & Module Organization
- `gear_sonic/`: Python package for SONIC teleoperation and sim utilities.
- `decoupled_wbc/`: Python package for decoupled whole-body control, plus `decoupled_wbc/tests/`.
- `gear_sonic_deploy/`: C++ deployment stack (TensorRT/ONNX, ROS2/ZMQ interfaces, unit tests).
- `docs/`: Sphinx docs source (`docs/source`) and build assets.
- `install_scripts/`: setup helpers (MuJoCo, ROS, Leap SDK, PICO).
- `external_dependencies/`: vendored third-party code; avoid local edits unless intentionally updating a dependency.

## Build, Test, and Development Commands
- Initial clone:
  - `git lfs pull` to fetch large assets and model files.
- Python environment (repo root):
  - `pip install -e "decoupled_wbc[dev]"`
  - `pip install -e "gear_sonic[teleop]"` (or `gear_sonic[sim]` for MuJoCo workflows)
- Lint/format:
  - `make run-checks` (`isort`, `black --check`, `ruff check`)
  - `make format` (auto-format Python)
  - `./lint.sh --fix` (CI-aligned lint autofix)
- Package build:
  - `make build` (builds Python distribution)
- Deploy stack build (C++):
  - `cd gear_sonic_deploy && just build`
- Docs build:
  - `pip install -r docs/requirements.txt && sphinx-build -b html docs/source docs/build/html`

## Coding Style & Naming Conventions
- Python: 4-space indentation; format with Black (line length 100), lint with Ruff, sort imports with isort.
- Python naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- C++ (`gear_sonic_deploy/`): `.clang-format` + `.editorconfig` enforce 2-space indentation, 120-column limit, C++20 targets.

## Testing Guidelines
- Primary Python test suite: `pytest decoupled_wbc/tests`.
- Keep tests near related modules under `decoupled_wbc/tests/<area>/` and name files `test_*.py`.
- C++ test sources live in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/`; run built binaries from `gear_sonic_deploy/target/release/` (for example `run_tests`).

## Commit & Pull Request Guidelines
- Follow existing history style: short, imperative commit subjects; optional prefixes like `[fix]`, `[add]`, `[del]` are acceptable.
- Keep each commit scoped to one logical change.
- PRs should include:
  - clear summary and motivation,
  - linked issue (if applicable),
  - validation steps/commands run,
  - platform details for deployment changes (x86_64 vs Jetson/JetPack),
  - screenshots/logs for docs or teleop UI behavior changes.
