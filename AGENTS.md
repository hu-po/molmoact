# Repository Guidelines

## Project Structure & Modules
- `olmo/`: Core Python package (models, data, training, eval). Start here.
- `launch_scripts/`: Entry points to train/eval (e.g., `train_multitask_model.py`).
- `scripts/`: Utilities (data downloads, conversion, generation). Examples: `download_data.py`, `download_robot_data.py`, `train.py`.
- `experiments/LIBERO/`: LIBERO assets and evaluation helpers.
- `assets/`: Images and logos.  
- `Dockerfile`: Reference environment used for training.

## Build, Test, and Development Commands
- Install (dev + train + serve): `pip install -e .[all]`
- Lint: `ruff check .`  | Format: `black .` and `isort .`
- Type check: `mypy olmo`
- Tests (quiet): `pytest -q` or `pytest -q olmo`
- Example data download: `python scripts/download_data.py all --n_procs 16`
- Example training (torchrun): see `launch_scripts/train_multitask_model.py` and README for mixtures and flags.

## Coding Style & Naming Conventions
- Python ≥3.10. Use type hints and docstrings for public functions.
- Indentation: 4 spaces; line length: 88–100 chars (black default is fine).
- Naming: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.
- Keep imports grouped: stdlib, third‑party, local; run `isort` and `black` before committing.

## Testing Guidelines
- Framework: `pytest`. Place tests near code or under `olmo/` with `test_*.py` filenames (e.g., `olmo/hf_model/molmoact/test_molmoact.py`).
- Add unit tests when changing model logic, data loaders, or launch scripts.
- Run `pytest -q` locally; include minimal fixtures and avoid network in tests.

## Commit & Pull Request Guidelines
- Commits: imperative, concise (e.g., "add LIBERO dependencies", "update imports"). Group related changes.
- PRs must include: purpose/summary, how to test (commands), linked issues, and any screenshots/logs for results.
- Ensure CI green: run `ruff`, `black`, `isort`, `mypy`, and `pytest` before requesting review.

## Security & Configuration Tips
- Set data/cache paths: `export MOLMOACT_DATA_DIR=/data/molmoact` and `export HF_HOME=/data/molmoact/huggingface`.
- Training/eval secrets: set `WANDB_API_KEY` via env vars; do not commit secrets.
- Offline safety during training: `export HF_DATASETS_OFFLINE=1`.
- VLLM eval on LIBERO may require: `export VLLM_WORKER_MULTIPROC_METHOD=spawn`.

