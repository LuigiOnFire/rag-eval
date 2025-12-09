#!/bin/bash
# Overnight pipeline: wait for generation, then train
# Run this after starting trajectory generation

LOG_DIR="/home/wcrawford/rag_eval/logs"
DATA_DIR="/home/wcrawford/rag_eval/data/processed"
VENV="/home/wcrawford/rag_eval/venv/bin/python"

echo "=== Overnight Pipeline Started at $(date) ===" >> $LOG_DIR/overnight_pipeline.log

# Wait for generation to complete
echo "Waiting for trajectory generation to complete..."
while pgrep -f "generate_trajectories.py" > /dev/null; do
    sleep 60
done

echo "=== Generation completed at $(date) ===" >> $LOG_DIR/overnight_pipeline.log

# Check if trajectories file exists
if [ ! -f "$DATA_DIR/trajectories_500.json" ]; then
    echo "ERROR: trajectories_500.json not found!" >> $LOG_DIR/overnight_pipeline.log
    exit 1
fi

# Show generation stats
echo "Generation stats:" >> $LOG_DIR/overnight_pipeline.log
$VENV -c "
import json
with open('$DATA_DIR/trajectories_500.json') as f:
    data = json.load(f)
correct = sum(1 for t in data if t.get('correct', False))
pairs = sum(len(t.get('trajectory', [])) for t in data if t.get('correct', False))
print(f'Total samples: {len(data)}')
print(f'Correct trajectories: {correct} ({100*correct/len(data):.1f}%)')
print(f'Training pairs: {pairs}')
" >> $LOG_DIR/overnight_pipeline.log 2>&1

# Train with class weighting
echo "=== Starting weighted training at $(date) ===" >> $LOG_DIR/overnight_pipeline.log
cd /home/wcrawford/rag_eval
$VENV scripts/train_and_test_epochs.py \
    --trajectories $DATA_DIR/trajectories_500.json \
    --output_dir models/controller_weighted_500 \
    --epochs 5 \
    --num_test_queries 25 \
    >> $LOG_DIR/overnight_training.log 2>&1

echo "=== Overnight Pipeline Completed at $(date) ===" >> $LOG_DIR/overnight_pipeline.log
echo "Check logs/overnight_training.log for training results"
