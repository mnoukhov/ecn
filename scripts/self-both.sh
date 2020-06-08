#!/bin/bash

python main.py \
    --name "self-prop-seed$SEED" \
    --savedir "$SLURM_TMPDIR" \
    --noprosocial \
    --proposal \
    --linguistic \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 50000 \
    --wandb \
    $@
