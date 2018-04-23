#!/bin/bash

source ~/.bashrc
source activate emerge
export PROJECTROOT="$HOME/iter-comm"
export PYTHONPATH=$PROJECTROOT:$PYTHONPATH

python src/ecn.py \
    --enable-cuda \
    --model-file 'model_saves/pro-ling.dat' \
    --name 'pro-ling' \
    --disable-proposal \
    --term-entropy-reg 0.5 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.01 \
    --render-every-seconds 120 \
    --save-every-seconds 360 \
    --episodes 300000 \
    $@
