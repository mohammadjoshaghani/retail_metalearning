#!/bin/bash

echo "welcome to the run!"
python ./src/experiment.py

echo "start syncing!"
wandb sync --sync-all
echo "syncing done!"
$SHELL