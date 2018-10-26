#!/bin/bash

python src/main.py \
    --name 'self-prop' \
    --noprosocial \
    --proposal \
    --nolinguistic \
    --enable_cuda \
    --model_dir "model_saves" \
    --logdir "logs" \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.005 \
    --render-every-episode 200 \
    --save-every-seconds 360 \
    --episodes 500000 \
