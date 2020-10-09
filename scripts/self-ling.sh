#!/bin/bash
python main.py \
    --name "self-ling-seed$SEED" \
    --savedir "$SCRATCH" \
    --noprosocial \
    --noproposal \
    --linguistic \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 50000 \
    --wandb \
    --wandb-offline \
    --seed $SEED \
    --device 'cuda' \
    $@
