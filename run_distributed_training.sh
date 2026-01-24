#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="default_config.yaml"
N_GPUS=3

torchrun --nproc_per_node $N_GPUS scripts.e2e_training \
    --config  "$SCRIPT_DIR/configs/$CONFIG_FILE"
