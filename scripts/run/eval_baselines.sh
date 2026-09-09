#!/usr/bin/env bash
# Evaluate the SFT and SFT-Think baselines trained by train_baselines.sh.
# `--test seen` reports action accuracy on the training split; `--evaluate` rolls out
# in the simulator, so VirtualHome must be running first (see externals/README.md)
# and ENV_IP / ENV_PORT must point at it.
set -euo pipefail

ENV=${ENV:-virtualhome}
CTYPE=${CTYPE:-behavior}
SEQ=${SEQ:-sequence_1}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
# stage order comes from the config so a sequence's stages are always evaluated in order
SETS=${SETS:-$(python3 -c "g={};exec(open('configs/$ENV/${CTYPE}_train.py').read(),g);print(' '.join(g['sequences']['$SEQ']))")}
ENV_IP=${ENV_IP:-127.0.0.1}
ENV_PORT=${ENV_PORT:-8080}

COMMON="--env_name $ENV --continual_type $CTYPE --continual_seq $SEQ \
        --pi_model_name $MODEL --wandb_off"

for s in $SETS; do
    python3 scripts/baselines/train_sft.py       $COMMON --continual_set "$s" --tag sft --test seen
    python3 scripts/baselines/train_sft.py       $COMMON --continual_set "$s" --tag sft \
        --evaluate both --env_ip "$ENV_IP" --env_port "$ENV_PORT"

    python3 scripts/baselines/train_sft_think.py $COMMON --continual_set "$s" --tag sft_think --test seen
    python3 scripts/baselines/train_sft_think.py $COMMON --continual_set "$s" --tag sft_think \
        --evaluate both --env_ip "$ENV_IP" --env_port "$ENV_PORT"
done
