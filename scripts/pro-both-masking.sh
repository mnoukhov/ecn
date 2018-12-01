#!/bin/bash

python src/main.py \
    --name 'pro-both-masking' \
    --prosocial \
    --proposal \
    --linguistic \
    --enable_cuda \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.05 \
    --episodes 100000 \
    --force_masking_comm \
    $@