#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="default_config.yaml"

python -m scripts.e2e_training\
    --config  "$SCRIPT_DIR/configs/$CONFIG_FILE"
