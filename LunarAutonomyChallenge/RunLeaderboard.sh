#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LEADERBOARD_ROOT="$SCRIPT_DIR/Leaderboard"
export TEAM_CODE_ROOT="$SCRIPT_DIR/agents"

export PYTHONPATH="$LEADERBOARD_ROOT:$TEAM_CODE_ROOT:$PYTHONPATH"

#export TEAM_AGENT="$SCRIPT_DIR/default/opencv_agent.py"
export TEAM_AGENT="$SCRIPT_DIR/my_agent.py"

export MISSIONS="$LEADERBOARD_ROOT/data/missions_training.xml"
export MISSIONS_SUBSET="0"

export CHECKPOINT_ENDPOINT="$SCRIPT_DIR/results"

export REPETITIONS="1"

export RECORD=1
export RECORD_CONTROL=1
export RESUME=

export QUALIFIER=1
export EVALUATION=
export DEVELOPMENT=

export SEED=0

source /opt/ros/humble/setup.bash

python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --missions="${MISSIONS}" \
  --missions-subset="${MISSIONS_SUBSET}" \
  --seed="${SEED}" \
  --repetitions="${REPETITIONS}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --record="${RECORD}" \
  --record-control="${RECORD_CONTROL}" \
  --resume="${RESUME}" \
  --qualifier="${QUALIFIER}" \
  --evaluation="${EVALUATION}" \
  --development="${DEVELOPMENT}"
