#!/usr/bin/env bash

export PROJECT=$HOME/ecn
export PYTHONPATH=$PYTHONPATH:$PROJECT

source activate ecn

"$@"
