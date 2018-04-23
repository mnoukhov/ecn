#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0-10:00

source ~/.bashrc
source activate emerge
export PROJECTROOT="$HOME/iter-comm"
export PYTHONPATH=$PROJECTROOT:$PYTHONPATH

name="self-prop"

python src/ecn.py \
    --enable-cuda \
    --model-file "model_saves/$name.dat" \
    --name $name \
    --disable-comms \
    --term-entropy-reg 0.05 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.005 \
    --render-every-seconds 120 \
    --save-every-seconds 360 \
    --episodes 500000 \
