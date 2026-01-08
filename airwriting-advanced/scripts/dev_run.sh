#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd)/src"
python -m airwriting_advanced.app.run_live --config config/config/config.yaml
