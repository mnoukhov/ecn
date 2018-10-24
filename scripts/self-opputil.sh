#!/bin/bash

python src/main.py \
    --name "self-opputil$1" \
    --noprosocial \
    --proposal \
    --linguistic \
    --force_utility_comm "$1" \
    --enable-cuda \
    --model_dir "model_saves" \
    --logdir "logs" \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.005 \
    --render-every-episode 200 \
    --save-every-seconds 360 \
    --episodes 500000 \
    "${@:2}"
