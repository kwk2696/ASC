#!/usr/bin/env bash
# SFT and SFT-Think baselines on the same continual sequence as train_asc.sh.
set -euo pipefail

ENV=${ENV:-virtualhome}
CTYPE=${CTYPE:-behavior}
SEQ=${SEQ:-sequence_1}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
# stage order comes from the config so a sequence's stages are always trained in order
SETS=${SETS:-$(python3 -c "g={};exec(open('configs/$ENV/${CTYPE}_train.py').read(),g);print(' '.join(g['sequences']['$SEQ']))")}

COMMON="--env_name $ENV --continual_type $CTYPE --continual_seq $SEQ \
        --pi_model_name $MODEL --wandb_off --train"

for s in $SETS; do
    python3 scripts/baselines/train_sft.py       $COMMON --continual_set "$s" --tag sft
    # SFT-Think trains two models: the rationale model (phase 1) and the action model (phase 2)
    python3 scripts/baselines/train_sft_think.py $COMMON --continual_set "$s" --tag sft_think --phase 1
    python3 scripts/baselines/train_sft_think.py $COMMON --continual_set "$s" --tag sft_think --phase 2
done
