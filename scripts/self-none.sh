#!/usr/bin/env bash
python main.py \
    --name "self-none-seed$SEED" \
    --savedir "$SCRATCH" \
    --noprosocial \
    --nolinguistic \
    --noproposal \
    --term-entropy-reg 0.05 \
    --proposal-entropy-reg 0.05 \
    --episodes 25000 \
    --wandb \
    --wandb-offline \
    --seed $SEED \
    --device 'cuda' \
    $@
