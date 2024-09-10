#!/bin/bash

MODEL_NAME="${1:-default}"
echo "Test model: $MODEL_NAME"

echo "Job started"
screen -dmS bz_pmhc_esm
screen -S bz_pmhc_esm -X stuff "source env/bin/activate; python main.py -m $MODEL_NAME > output.log 2>&1; deactivate; exit\n"
