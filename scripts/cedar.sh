#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0-18:00
#SBATCH --account=rrp-bengoiy
#SBATCH --exclude=cdr338,cdr351

export PYTHONUNBUFFERED=1
export PROJECT=$HOME/ecn
export PYTHONPATH=$PYTHONPATH:$PROJECT

source ~/.bashrc
source activate ecn

"$@"
