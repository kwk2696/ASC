# Amortizing Self-Correction for Continual Embodied Adaptation

Code for *Amortizing Self-Correction for Continual Embodied Adaptation*
(Woo Kyung Kim, Jooyong Kim, Minjong Yoo, Honguk Woo; AACL-IJCNLP 2026).

## Overview

ASC trains two LoRA adapters on a frozen LLM:

- `reasoning_policy` writes a five-slot rationale and learns to correct the base
  policy's rationale as well as its own.
- `planning_policy` maps the rationale to an action.

Each stage of a continual sequence starts from the earlier stage whose rationales fit the
new data best, and corrects that stage's policy. At inference, `--test_time_control` adds
rationale slots one at a time and stops once the planner's confidence passes a calibrated
threshold (`--test_time_level 0-6`, lower stops earlier).

## Installation

```bash
git clone https://github.com/kwk2696/ASC.git && cd ASC
pip install -r requirements.txt
pip install -e . -e externals/virtualhome -e externals/vh_dataset
bash tools/setup_externals.sh        # SKIP_ALFRED=1 for VirtualHome only
```

Run everything from the repository root. `tools/setup_externals.sh` fetches CL-ALFRED and
the ALFRED data; the VirtualHome Unity executable is a separate download needed only for
`--evaluate` (see [externals/README.md](externals/README.md)). Dataset annotation calls
the OpenAI API and needs `OPENAI_API_KEY`.

## Data

| Path | Contents |
|------|----------|
| `datasets/{virtualhome,alfred}/{behavior,environment}_train/set_*/` | `full_dataset.jsonl` (all annotated steps), `part_1.0_dataset.jsonl` (default training subset: 12.5% of full on VirtualHome, 25% on ALFRED), `part_2.0` (2x the default) and, on VirtualHome only, `part_4.0` (4x) |
| `datasets/{virtualhome,alfred}/{behavior,environment}_{seen,unseen}/set_*/` | evaluation trajectories |

`behavior` and `environment` are the two continual settings. Stage order per sequence is
in `configs/<env>/<type>_train.py`.

To rebuild from the simulator (`scripts/dataset/test_virtualhome.py` checks that the simulator
is reachable; `test_alfred.py` lists the ALFRED game files per split). Step 1 writes the raw
episodes to `<set>/raw/` (not shipped) and the evaluation episodes to `expert_traj.jsonl`:

```bash
python3 scripts/dataset/generate_virtualhome.py --type environment_train --episodes 2 --ip 127.0.0.1
python3 scripts/dataset/annotate_dataset.py --env_name virtualhome \
    --dataset_dir ./datasets/virtualhome/environment_train/set_1
python3 scripts/dataset/resize_dataset.py --env_name virtualhome \
    --continual_type environment --continual_set set_1 --dataset_scale 1.0
```

## Training and evaluation

```bash
ENV=virtualhome CTYPE=behavior SEQ=sequence_1 bash scripts/run/train_asc.sh
ENV=virtualhome CTYPE=behavior SEQ=sequence_1 bash scripts/run/eval_asc.sh
```

Or per stage:

```bash
python3 scripts/asc/train.py --env_name virtualhome \
    --continual_type behavior --continual_seq sequence_1 --continual_set set_1 \
    --pi_model_name Qwen/Qwen2.5-0.5B --tag release --train --wandb_off
```

- `--test 1` calibrates the per-slot confidence thresholds (`thresholds.json`); run it
  before `--evaluate` with `--test_time_control`.
- `--evaluate seen|unseen|both` rolls out in the simulator (`--env_ip`, `--env_port`).
- Results go under `results/`; `scripts/asc/check.py` prints them per sequence.
- W&B project `ASC`; disable with `--wandb_off`.

## Baselines

```bash
ENV=virtualhome CTYPE=behavior SEQ=sequence_1 bash scripts/run/train_baselines.sh
ENV=virtualhome CTYPE=behavior SEQ=sequence_1 bash scripts/run/eval_baselines.sh
```

SFT (`scripts/baselines/train_sft.py`) and SFT-Think (`train_sft_think.py`) take the same
arguments as `scripts/asc/train.py`. SFT-Think is trained in two calls per stage:
`--phase 1` (rationale model) then `--phase 2` (action model). `--test seen` reports action
accuracy; `--evaluate` rolls out in the simulator.

## Repository layout

```
embodied_cd/agents/asc/    ASC agent
embodied_cd/trl/asc/       ASC trainer
embodied_cd/environments/  VirtualHome and ALFRED environments
embodied_cd/common/        prompts, dataset loading, CLI
scripts/asc/               train / test / evaluate entry point, results reader
scripts/baselines/         SFT and SFT-Think
scripts/dataset/           dataset generation, annotation, subsetting
scripts/run/               example scripts
tools/                     third-party data setup
configs/                   task splits and prompts
datasets/                  annotated data
externals/                 VirtualHome package and third-party setup
```

## Citation

```bibtex
@inproceedings{kim2026amortizing,
  title     = {Amortizing Self-Correction for Continual Embodied Adaptation},
  author    = {Kim, Woo Kyung and Kim, Jooyong and Yoo, Minjong and Woo, Honguk},
  booktitle = {Proceedings of the Asia-Pacific Chapter of the Association for Computational
               Linguistics and the International Joint Conference on Natural Language
               Processing (AACL-IJCNLP)},
  address   = {Hengqin, China},
  month     = nov,
  year      = {2026}
}
```

## License

MIT. Builds on [VirtualHome](https://github.com/xavierpuigf/virtualhome) (MIT, bundled
under `externals/virtualhome`), [ALFWorld](https://github.com/alfworld/alfworld) (MIT),
[CL-ALFRED](https://github.com/snumprlab/cl-alfred) (GPL-3.0, fetched at install time)
and [TRL](https://github.com/huggingface/trl) (Apache-2.0).
