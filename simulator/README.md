# Installation

```bash
git submodule update --init --recursive third_party/RoboLab

cd simulator
uv venv --python 3.11
uv sync
```

`setuptools<81` is pinned via `constraint-dependencies` in `pyproject.toml`
because `flatdict` (transitive dep of `isaaclab`) imports `pkg_resources` at
build time, which was removed in setuptools 81.

## First-run EULA

Isaac Sim prompts for EULA acceptance on first import. Either run once in an
interactive shell and type `Yes`, or export:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Verify the install by listing RoboLab's registered tasks. Isaac Sim prompts interactively for EULA acceptance on first boot — set OMNI_KIT_ACCEPT_EULA=YES to accept non-interactively (required on every invocation):

```bash
uv run python ../third_party/RoboLab/scripts/check_registered_envs.py
```

## Server + client rollout

Server (in `openpi_GUIDANCE/`, pin to its own GPU — JAX pre-allocation will
crash Isaac Sim if they share):

```bash
cd /home/asethi04/ROBOTICS/openpi_GUIDANCE
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py \
    --port 8001 policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/home/asethi04/.cache/openpi/openpi-assets/checkpoints/pi05_droid
```

Client (in `simulator/`, once server prints `server listening on 0.0.0.0:8001`).
Interactive by default — prompts for instruction (empty = task default),
success (y/n), and whether to run another:

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=3 \
  .venv/bin/python -u main_robolab.py \
      --task BananaOnPlateTask \
      --remote-host 127.0.0.1 --remote-port 8001 \
      --max-timesteps 600 --headless --save-dir runs
```

To run non-interactively (one rollout, default instruction, auto-exit), pipe
the three prompts via stdin:

```bash
printf '\ny\nn\n' | OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=3 \
  .venv/bin/python -u main_robolab.py \
      --task BananaOnPlateTask \
      --remote-host 127.0.0.1 --remote-port 8001 \
      --max-timesteps 600 --headless --save-dir runs
```

Smoke test with no server:

```bash
printf '\ny\nn\n' | OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=3 \
  .venv/bin/python -u main_robolab.py \
      --task BananaOnPlateTask --fake-policy --max-timesteps 3 --headless
```

Tests:

```bash
.venv/bin/python -m pytest tests/
```