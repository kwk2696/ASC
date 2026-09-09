#!/usr/bin/env bash
# Train ASC on one continual sequence, stage by stage. Each stage after the first
# picks the earlier stage whose rationales fit the new data best as its base policy.
set -euo pipefail

ENV=${ENV:-virtualhome}
CTYPE=${CTYPE:-behavior}          # behavior | environment
SEQ=${SEQ:-sequence_1}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
TAG=${TAG:-release}
# stage order comes from the config so a sequence's stages are always trained in order
SETS=${SETS:-$(python3 -c "g={};exec(open('configs/$ENV/${CTYPE}_train.py').read(),g);print(' '.join(g['sequences']['$SEQ']))")}

COMMON="--env_name $ENV --continual_type $CTYPE --continual_seq $SEQ \
        --pi_model_name $MODEL --tag $TAG --wandb_off"

for s in $SETS; do
    echo "=== $s ==="
    python3 scripts/asc/train.py $COMMON --continual_set "$s" --train
done
