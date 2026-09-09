# Third-party dependencies

## Bundled in this repository

| Path | What it is | License |
|------|------------|---------|
| `externals/virtualhome/` | The VirtualHome python package (`pip install -e externals/virtualhome`), taken from https://github.com/xavierpuigf/virtualhome (package version 2.3.0, `master` as of November 2024) with this project's modifications applied. Simulator assets are not included. | MIT, see `externals/virtualhome/LICENSE` |
| `externals/vh_dataset/` | VirtualHome knowledge-graph and expert-policy utilities written for ASC (`pip install -e externals/vh_dataset`). First-party. | MIT (this repository) |
| `externals/alfworld/configs/` | ALFWorld base configuration, needed to construct the ALFRED environment. | MIT (ALFWorld) |
| `externals/alfworld/object_properties.json` | Object-property table for ALFRED written for ASC. | MIT (this repository) |

This project's changes to VirtualHome are in `virtualhome/simulation/environment/resources.py`
(task and goal tables), `virtualhome/simulation/environment/unity_environment.py` (step and
observation handling), `virtualhome/simulation/unity_simulator/comm_unity.py` (request timeout),
the package `__init__.py`, and a headless launcher in `docker/run_sim.sh`.

## Fetched by `tools/setup_externals.sh`

Run once from the repository root. `SKIP_ALFRED=1` skips the ALFRED task data for a
VirtualHome-only setup.

```bash
bash tools/setup_externals.sh
```

| Path | Upstream | License |
|------|----------|---------|
| `externals/cl-alfred/embodied_split/` | https://github.com/snumprlab/cl-alfred, sparse checkout of the continual splits only (the full repository carries ~10GB of data) | GPL-3.0 |
| `externals/alfworld/json_2.1.1/` | https://github.com/alfworld/alfworld, via `alfworld-download` | MIT |

`cl-alfred` is GPL-3.0 licensed, which is why it is fetched rather than redistributed;
keeping it out of this tree lets this repository stay under the MIT license in `LICENSE`.

## Simulator binary

VirtualHome needs the Unity executable, a separate download that is only required for
simulator rollouts (`--evaluate`), not for training:

```bash
wget http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip
unzip linux_exec.zip -d ~/virtualhome
bash externals/virtualhome/docker/run_sim.sh   # headless, serves on --http-port
```
