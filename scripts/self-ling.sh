#!/bin/bash

python src/main.py \
    --name 'self-ling' \
    --savedir "$HOME" \
    --noprosocial \
    --noproposal \
    --linguistic \
    --term-entropy-reg 0.2 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.005 \
    --episodes 250000 \
    --wandb \
    $@
