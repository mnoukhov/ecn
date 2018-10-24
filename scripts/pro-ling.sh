#!/bin/bash

python src/main.py \
    --name 'pro-ling' \
    --prosocial \
    --noproposal \
    --linguistic \
    --enable-cuda \
    --model_dir 'model_saves' \
    --logdir 'logs' \
    --term-entropy-reg 0.5 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.01 \
    --render-every-episode 200 \
    --save-every-seconds 360 \
    --episodes 300000 \
    $@
