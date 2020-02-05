#!/usr/bin/env bash

HASH_ID=$(docker inspect --format="{{.Id}}" volatile-images.borgy.elementai.net/mnoukhov/ecn:latest)
DOCKER_ID=${HASH_ID:7}

borgy submit \
    --gpu 0 \
    --mem 4 \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_DOCKER=$DOCKER_ID \
    -e HOME=/workdir \
    -i volatile-images.borgy.elementai.net/mnoukhov/ecn:latest \
    -v /home/mnoukhov/ecn:/workdir/ecn:ro \
    -- bash -c "cd ecn; source ./scripts/local.sh; $@"
