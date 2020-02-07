#!/usr/bin/env bash

python src/main.py \
    --name "self-none-seed$SEED" \
    --savedir "$HOME" \
    --noprosocial \
    --nolinguistic \
    --noproposal \
    --term-entropy-reg 0.05 \
    --proposal-entropy-reg 0.05 \
    --episodes 50000 \
    --wandb \
    --seed $SEED \
    $@
