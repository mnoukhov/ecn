#!/bin/bash

python src/main.py \
    --name "self-prop-seed$SEED" \
    --savedir "$HOME" \
    --noprosocial \
    --proposal \
    --nolinguistic \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 50000 \
    --wandb \
    --seed $SEED \
    $@
