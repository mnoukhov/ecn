#!/bin/bash

python src/main.py \
    --name 'self-prop' \
    --noprosocial \
    --proposal \
    --nolinguistic \
    --enable_cuda \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.005 \
    --episodes 250000 \
    $@
