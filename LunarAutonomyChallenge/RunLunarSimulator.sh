#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SIMULATOR_ROOT="$SCRIPT_DIR/LunarSimulator"

bash "$SIMULATOR_ROOT/LAC.sh"
