#!/bin/bash
#SBATCH --job-name=il-gotoobj-guide
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=60000M
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

# Sample script for training a single Guide

ENV='BabyAI-GoToObj-v0'
DEMOS='RL-experts/best/gotoobj-1k'
BATCH_SIZE=128
IMAGE_DIM=256
MEMORY_DIM=2048
INSTR_DIM=256
CORR_LENGTH=2
CORR_VOCAB_SIZE=3
SEED=0
VAL_EPISODES=500
LEARNING_RATE=1e-4
DROPOUT=0.5

python3 scripts/train_il.py --corrector --corr-own-vocab \
    --env $ENV --demos $DEMOS --batch-size $BATCH_SIZE \
    --image-dim $IMAGE_DIM --memory-dim $MEMORY_DIM \
    --instr-dim $INSTR_DIM --corr-length $CORR_LENGTH \
    --corr-vocab-size $CORR_VOCAB_SIZE \
    --seed $SEED --val-episodes $VAL_EPISODES \
    --lr $LEARNING_RATE --dropout $DROPOUT