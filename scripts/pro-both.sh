#!/bin/bash

source ~/.bashrc
source activate emerge
export PROJECTROOT="~/iter-comm"
export PYTHONPATH=$PROJECTROOT:$PYTHONPATH

python src/ecn.py \
    --enable-cuda \
    --model-file 'model_saves/pro-both.dat' \
    --name 'pro-both' \
    --term-entropy-reg 0.5 \
    --utterance-entropy-reg 1e-4 \
    --proposal-entropy-reg 0.01 \
    --render-every-seconds 120 \
    --save-every-seconds 360 \
    --episodes 3e5