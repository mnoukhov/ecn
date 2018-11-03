#!/bin/bash

python src/main.py \
    --name 'pro-ling-masking' \
    --prosocial \
    --noproposal \
    --linguistic \
    --enable_cuda \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 250000 \
    --force_masking_comm \
    $@
