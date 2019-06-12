#!/usr/bin/env python3

"""
Check relation between messages and input

example usage:

python3 scripts/message_input_dist.py --env BabyAI-GoToObj-v0 --model gotoobj/gotoobj-cor-ownvocab1 --episodes 10 --plot

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
import itertools

import babyai.utils as utils
from babyai.evaluate import evaluate_demo_agent, evaluate, ManyEnvs
from interpret_obs import interpret_obs

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
        "corrections_per_episode": [],
        "observed_facts" : []
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
        obs_facts = [[] for _ in range(num_envs)]

        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():
            result = agent.act_batch(many_obs)

            action = result['action']
            correction = result['corr']

            # slice 0: objects, slice 1: colors, slice 2: locked
            obj_images = [(np.flip(obs['image'][:, :, 0], 0), obs['mission']) for obs in many_obs]
            observed_facts = interpret_obs(many_obs)

            for _ in range(num_envs):
                if not already_done[_]:
                    corrections[_].append(correction[_])
                    obs_facts[_].append(observed_facts[_])
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
        logs["observed_facts"].extend(list(obs_facts))
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
    obs_facts = logs['observed_facts']

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

        OBSERVABLE_FACTS = ['object_ahead', 'object_immediately_left', 'object_immediately_right', 'object_invisible']
        CORRS = list(itertools.product(['w0', 'w1', 'w2'], repeat=2)) # assuming two-word corrections, alphabet size 3
        # plotting, assuming two-word corrections
        d = {}

        for of in OBSERVABLE_FACTS:
            d[of] = [facts[of] for episode_facts in obs_facts for facts in episode_facts]

        d['corr'] = [correction for episode_corrections in corrections for correction in episode_corrections]
        df = pd.DataFrame(d)

        for idx, of in enumerate(OBSERVABLE_FACTS):
            df_segment = df.loc[df[of]]
            corrs = [str(tuple(c)) for c in df_segment['corr'].tolist()]

            c = Counter()
            c.update({corr: 0 for corr in [str(item) for item in CORRS]})
            c.update(corrs)
            labels, values = zip(*c.items())
            count = sum(values)
            values = [value / count for value in values]
            indexes = np.arange(1, len(labels) + 1)
            width = 0.5
            plt.figure(figsize=(8, 2))
            plt.bar(indexes, values, width, color='#66c2a5')
            plt.xticks(indexes, labels=[c[0] + c[1] for c in CORRS])
            plt.savefig(args.env + of + '.pdf')
            plt.clf()