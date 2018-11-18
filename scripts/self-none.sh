#!/bin/bash

python src/main.py \
    --name "self-none" \
    --noprosocial \
    --nolinguistic \
    --noproposal \
    --enable_cuda \
    --term-entropy-reg 0.2 \
    --proposal-entropy-reg 0.005 \
    --episodes 250000 \
    $@
