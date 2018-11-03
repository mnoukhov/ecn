#!/bin/bash

python src/main.py \
    --name 'self-ling' \
    --noprosocial \
    --noproposal \
    --linguistic \
    --enable_cuda \
    --term-entropy-reg 0.2 \
    --utterance-entropy-reg 0.001 \
    --proposal-entropy-reg 0.005 \
    --episodes 250000 \
    $@
