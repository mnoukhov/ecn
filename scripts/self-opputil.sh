#!/bin/bash

source ~/.bashrc
source activate emerge
export PROJECTROOT="$HOME/iter-comm"
export PYTHONPATH=$PROJECTROOT:$PYTHONPATH

name="self-opputil$1"

python src/ecn.py \
    --enable-cuda \
    --model-file "model_saves/$name.dat" \
    --name $name \
    --selfish \
    --enable-comms \
    --comms-opponent-utility $1 \
    --enable-proposal \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.005 \
    --render-every-seconds 120 \
    --save-every-seconds 360 \
    --episodes 500000 \
