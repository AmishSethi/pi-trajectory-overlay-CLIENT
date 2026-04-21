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