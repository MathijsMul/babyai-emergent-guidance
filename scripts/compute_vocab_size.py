"""
Compute vocabulary size for given level
"""

import argparse

import babyai
import gym
import re

def compute_vocab_size(level, n_episodes=1000):
    env = gym.make(level)
    instructions = set(env.reset()['mission'] for i in range(n_episodes))

    level_words = set()
    for instr in sorted(instructions):
        tokens = re.findall("([a-z]+)", instr.lower())
        for word in tokens:
            level_words.add(word)

    num_words = len(level_words)
    return(num_words)

# n = compute_vocab_size('BabyAI-BossLevel-v0')

def compute_sizes(level_file, out_file):
    levels = open(level_file).readlines()
    with open(out_file, 'w') as f:
        for level in levels:
            level = level.strip()
            f.write(level + '\t' + str(compute_vocab_size(level)) + '\n')

compute_sizes('babyai_levels.txt', 'babyai_vocab_sizes.txt')