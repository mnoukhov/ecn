#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export PROJECT=$HOME/ecn
export PYTHONPATH=$PYTHONPATH:$PROJECT

source ~/.bashrc
source activate ecn

"$@"
