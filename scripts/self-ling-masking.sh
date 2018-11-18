#!/bin/bash

python src/main.py \
    --name 'self-ling-masking' \
    --noprosocial \
    --noproposal \
    --linguistic \
    --enable_cuda \
    --term-entropy-reg 0.1 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.005 \
    --episodes 250000 \
    --force_masking_comm \
    $@
