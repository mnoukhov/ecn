#!/bin/bash

python src/main.py \
    --name "self-ling-masking-seed$SEED" \
    --savedir "$HOME" \
    --noprosocial \
    --noproposal \
    --linguistic \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 50000 \
    --force_masking_comm \
    --wandb \
    --seed $SEED \
    $@
