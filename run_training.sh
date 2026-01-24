#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="default_config.yaml"

python -m scripts.train_backup\
    --config  "$SCRIPT_DIR/configs/$CONFIG_FILE"
