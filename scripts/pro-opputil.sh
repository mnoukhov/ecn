#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0-10:00

source ~/.bashrc
source activate emerge
export PROJECTROOT="$HOME/iter-comm"
export PYTHONPATH=$PROJECTROOT:$PYTHONPATH

name="pro-opputil$1"

python src/ecn.py \
    --enable-cuda \
    --model-file "model_saves/$name.dat" \
    --name $name \
    --disable-comms \
    --enable-opponent-utility $1 \
    --term-entropy-reg 0.5 \
    --utterance-entropy-reg 0.0001 \
    --proposal-entropy-reg 0.01 \
    --render-every-seconds 120 \
    --save-every-seconds 360 \
    --episodes 500000 \
