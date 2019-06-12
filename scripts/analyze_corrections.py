#!/usr/bin/env python3

"""
Check relation between messages and actions

example usage:

python3 scripts/analyze_corrections.py --env BabyAI-GoToLocal-v0 --model gotolocal/pretrainedcor-cic-allepoch/BabyAI-GoToLocal-v0_IL_pretrainedcorr_own-vocab_expert_filmcnn_gru_seed8491_19-05-03-15-55-42_epoch3 --episodes 500 --plot

"""

import argparse
import gym
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import babyai.utils as utils
from babyai.evaluate import evaluate_demo_agent, evaluate, ManyEnvs

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent) -- needs to be set to 0 if valid")
parser.add_argument("--argmax", action="store_true", default=True,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--contiguous-episodes", action="store_true", default=False,
                    help="Make sure episodes on which evaluation is done are contiguous")
parser.add_argument("--plot", action="store_true", default=False,
                    help="Plot correction/action statistics")


def main(args, seed, episodes):
    # Set seed for all randomness sources
    utils.seed(seed)

    # Define agent

    env = gym.make(args.env)
    env.seed(seed)
    agent = utils.load_agent(env, args.model, None, None, args.argmax, args.env)
    if args.model is None and args.episodes > len(agent.demos):
        # Set the number of episodes to be the number of demos
        episodes = len(agent.demos)

    # Evaluate
    if isinstance(agent, utils.DemoAgent):
        logs = evaluate_demo_agent(agent, episodes)
    elif isinstance(agent, utils.BotAgent) or args.contiguous_episodes:
        logs = evaluate(agent, env, episodes, False)
    else:
        logs = batch_evaluate(agent, args.env, seed, episodes, return_obss_actions=True)

    return logs

# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, seed_shift=1e9, return_obss_actions=False):
    num_envs = min(256, episodes)

    seed += seed_shift
    seed = int(seed)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": [],
        "corrections_per_episode": []
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)
        env.seed(seeds)

        many_obs = env.reset()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')

        corrections = [[] for _ in range(num_envs)]

        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():

            result = agent.act_batch(many_obs, cur_num_frames)
            action = result['action']
            correction = result['corr']

            for _ in range(num_envs):
                if not already_done[_]:
                    corrections[_].append(correction[_])
                    if return_obss_actions:
                        obss[_].append(many_obs[_])
                        actions[_].append(action[_].item())

            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True
        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        logs["corrections_per_episode"].extend(list(corrections))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

    return logs

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 0 if args.model is not None else 1

    start_time = time.time()
    logs = main(args, args.seed, args.episodes)
    corrections = logs['corrections_per_episode']
    actions = logs['actions_per_episode']

    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)

    if args.model is not None:
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])

    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    if args.model is not None:
        print("F {} | FPS {:.0f} | D {} | R:xsmM {:.2f} {:.2f} {:.2f} {:.2f} | S {:.2f} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration,
                      *return_per_episode.values(),
                      success_per_episode['mean'],
                      *num_frames_per_episode.values()))
    else:
        print("F {} | FPS {:.0f} | D {} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration, *num_frames_per_episode.values()))

    if args.plot:
        action_dict = {0: 'left',
                       1: 'right',
                       2: 'forward',
                       3: 'pickup',
                       4: 'drop',
                       5: 'toggle',
                       6: 'done'}

        # plotting, assuming two-word corrections
        d = {}
        d['word1'] = [word[0] for episode_corrections in corrections for word in episode_corrections]
        d['word2'] = [word[1] for episode_corrections in corrections for word in episode_corrections]
        d['action'] = [action for episode_actions in actions for action in episode_actions]
        df = pd.DataFrame(d)

        word1_values = set(df['word1'])
        word2_values = set(df['word2'])
        action_values = set(df['action'])

        num_actions = len(action_values)

        fig, axes = plt.subplots(len(word2_values), len(word1_values), sharex='col', sharey='row')#
        plt.rc('font', size=13)  # 13

        plt.suptitle('Some-title')

        for idx1, word1 in enumerate(sorted(word1_values)):
            for idx2, word2 in enumerate(sorted(word2_values)):
                try:
                    df_segment = df.loc[(df['word1'] == word1) & (df['word2'] == word2)]
                    actions = df_segment['action'].tolist()

                    c = Counter()
                    c.update({action: 0 for action in action_values})
                    c.update(actions)
                    labels, values = zip(*c.items())
                    count = sum(values)
                    values = [value / count for value in values]

                    indexes = np.arange(1, len(labels) + 1)
                    width = 0.5

                    ax = axes[idx2, idx1]
                    ax.bar(indexes, values, width, color='#66c2a5')
                    ax.set_xticks(indexes)

                except:
                    pass

        cols = sorted(word1_values)
        rows = sorted(word2_values)

        for ax, col in zip(axes[-1], cols):
            ax.set_xlabel(col)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, labelpad=10)

        fig.text(0.5, 0.0, 'word 1', transform=fig.transFigure)
        fig.text(0.0, 0.5, 'word 2', transform=fig.transFigure, rotation='vertical')

        #plt.show()
        plt.savefig('plotname.pdf')