#!/usr/bin/env python3

"""
Observe agent behavior when sending scripted guidance messages
"""

import argparse
import gym
import time
import numpy as np
import random
from collections import Counter

import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")

args = parser.parse_args()

action_map = {
    "LEFT"   : "left",
    "RIGHT"  : "right",
    "UP"     : "forward",
    "PAGE_UP": "pickup",
    "PAGE_DOWN": "drop",
    "SPACE": "toggle"
}

assert args.model is not None or args.demos_origin is not None, "--model or --demos-origin must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# Define agent
agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)

# If we want to replace Corrector with a scripted one

# gotoobj
message_space = [['w0', 'w0'], ['w1', 'w1'], ['w2', 'w2']]
# waltz = [[['w0', 'w0']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w2', 'w2']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']], [['w0', 'w0']], [['w1', 'w1']],
#           [['w1', 'w1']]]
#pirouette = [[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']],[['w0', 'w0']]]
# expected_corr = {str(['w0', 'w0']) : 1,
#                  str(['w1', 'w1']) : 2,
#                  str(['w2', 'w2']) : 0}

# gotolocal (use --seed 13)
# message_space = [['w2', 'w0'], ['w0', 'w1'], ['w1', 'w2'], ['w0', 'w0'], ['w1', 'w1'], ['w2', 'w2']]
#pirouette = [[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']],[['w0', 'w1']]]
# pirouette = [[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],[['w1', 'w1']],]
# waltz = [[['w0', 'w1']],
#          [['w0', 'w1']], [['w1', 'w2']],[['w1', 'w2']],[['w1', 'w2']],
#          [['w1', 'w1']], [['w1', 'w2']], [['w1', 'w2']],
#          [['w1', 'w1']], [['w1', 'w2']], [['w1', 'w2']],
#          [['w0', 'w1']], [['w1', 'w2']], [['w1', 'w2']],
#          [['w1', 'w1']], [['w1', 'w2']], [['w1', 'w2']],
#         [['w1', 'w1']], [['w1', 'w2']], [['w1', 'w2']],
#         [['w1', 'w1']], [['w1', 'w2']], [['w1', 'w2']]
#          ]
# expected_corr = {str(['w2', 'w0']) : 0,
#                  str(['w0', 'w1']) : 1,
#                  str(['w1', 'w2']) : 2,
#                  str(['w0', 'w0']): 3,
#                  str(['w1', 'w1']): 1,
#                  str(['w2', 'w2']): 2}

random_messages = [[random.choice(message_space)] for i in range(500)]

# exp_corr = {['w0, w0'] : Counter(),
#             ['w1, w1'] : 2,
#             ['w2, w2'] : 0}
#exp_corr = {message : Counter() for message in message_space}

script = random_messages
agent.set_corrector_script(script = script)

# Run the agent
done = True

action = None

def keyDownCb(keyName):
    global obs
    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in action_map and keyName != "RETURN":
        return

    agent_action = agent.act(obs)['action']

    if keyName in action_map:
        action = env.actions[action_map[keyName]]

    elif keyName == "RETURN":
        action = agent_action

    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)
    if done:
        print("Reward:", reward)
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))

actions = []
total_step = 0
step = 0
while True:
    time.sleep(args.pause)
    renderer = env.render("human")
    if args.manual_mode and renderer.window is not None:
        renderer.window.setKeyDownCb(keyDownCb)

    else:
        try:
            result = agent.act(obs, time=total_step)
        except:
            message_actions = {str(message): Counter() for message in message_space}
            for idx, item in enumerate(script):
                message_actions[str(item[0])][actions[idx]] += 1

            # compute obedience
            gold, total = 0, 0
            for key in message_actions:
                expected_action = expected_corr[key]
                total_key = sum(message_actions[key].values())
                gold_key = message_actions[key][expected_action]
                total += total_key
                gold += gold_key

            obedience = gold / total
            print('Obedience:')
            print(obedience)
            exit()
        print('action: ' + str(result['action'].item()))
        actions.append(result['action'].item())
        obs, reward, done, _ = env.step(result['action'])

        agent.analyze_feedback(reward, done)
        if 'dist' in result and 'value' in result:
            dist, value = result['dist'], result['value']
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            if value is None:
                print("step: {}, mission: {}, dist: {}, entropy: {:.2f}".format(
                    step, obs["mission"], dist_str, float(dist.entropy())))
            else:
                print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                    step, obs["mission"], dist_str, float(dist.entropy()), float(value)))
        else:
            print("step: {}, mission: {}".format(step, obs['mission']))

        if done:
            print("Reward:", reward)
            obs = env.reset()
            agent.on_reset()
            step = 0
        else:
            step += 1

        total_step += 1
        time.sleep(1)

    if renderer.window is None:
        break
