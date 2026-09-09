#!/usr/bin/env bash
# Evaluate a trained ASC checkpoint. `--test` calibrates the per-slot confidence
# thresholds on the training split (needed by test-time control); `--evaluate` rolls
# out in the simulator, so VirtualHome must be running first
# (see externals/README.md) and ENV_IP / ENV_PORT must point at it.
set -euo pipefail

ENV=${ENV:-virtualhome}
CTYPE=${CTYPE:-behavior}
SEQ=${SEQ:-sequence_1}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
TAG=${TAG:-release}
# stage order comes from the config so a sequence's stages are always trained in order
SETS=${SETS:-$(python3 -c "g={};exec(open('configs/$ENV/${CTYPE}_train.py').read(),g);print(' '.join(g['sequences']['$SEQ']))")}
ENV_IP=${ENV_IP:-127.0.0.1}
ENV_PORT=${ENV_PORT:-8080}

COMMON="--env_name $ENV --continual_type $CTYPE --continual_seq $SEQ \
        --pi_model_name $MODEL --tag $TAG --wandb_off"

for s in $SETS; do
    python3 scripts/asc/train.py $COMMON --continual_set "$s" --test 1

    python3 scripts/asc/train.py $COMMON --continual_set "$s" \
        --evaluate both --test_time_control --test_time_level 1 \
        --env_ip "$ENV_IP" --env_port "$ENV_PORT"
done
