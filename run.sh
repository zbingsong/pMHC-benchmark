#!/bin/bash

MODEL_NAME="${1:-default}"
echo "Test model: $MODEL_NAME"
SCREEN_NAME="pmhc_benchmark_$MODEL_NAME"

echo "Job started"
screen -dmS "$SCREEN_NAME"
screen -S "$SCREEN_NAME" -X stuff "env/bin/python main.py -m $MODEL_NAME > logs/output_$MODEL_NAME.log 2>&1; exit\n"
