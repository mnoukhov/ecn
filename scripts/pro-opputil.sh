#!/bin/bash

python src/main.py \
    --name "pro-opputil$1" \
    --prosocial \
    --proposal \
    --linguistic \
    --force_utility_comm $1 \
    --enable_cuda \
    --model_dir "model_saves" \
    --logdir "logs" \
    --term-entropy-reg 0.5 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.01 \
    --render-every-episode 200 \
    --save-every-seconds 360 \
    --episodes 500000 \
    "${@:2}"
