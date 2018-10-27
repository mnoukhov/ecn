#!/bin/bash

python src/main.py \
    --name 'pro-both' \
    --prosocial \
    --proposal \
    --linguistic \
    --enable_cuda \
    --term-entropy-reg 0.5 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.01 \
    --episodes 250000 \
    $@
