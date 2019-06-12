#!/bin/bash
#SBATCH --job-name=il-gotoobj-nocor
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu

# Sample script for training a single Learner

python3 scripts/train_il.py --env BabyAI-GoToObj-v0 --demos RL-experts/best/gotoobj-1k --batch-size 128 --image-dim 256 --memory-dim 2048 --instr-dim 256 --seed 0 --learner