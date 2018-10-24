#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0-10:00

export PYTHONUNBUFFERED=1
export PROJECT=$HOME/ecn
export PYTHONPATH=$PYTHONPATH:$PROJECT

source ~/.bashrc
source activate ecn

"$@"
