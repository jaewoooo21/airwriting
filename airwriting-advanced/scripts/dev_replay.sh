#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/src"
python -m airwriting_advanced.app.replay_csv --config config/config/config.yaml --csv "$1"
