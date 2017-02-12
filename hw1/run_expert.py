#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v1 --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from gym import wrappers
import load_policy
import time

FPS = 10.0

SECONDS_PER_ITER = 1.0 / FPS


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy('experts/{}.pkl'.format(args.envname))
    print('loaded and built')

    last_time = 0.0
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        env = wrappers.Monitor(env, '/tmp/{}'.format(args.envname), force=True)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if True:
                    while time.time() - last_time <= SECONDS_PER_ITER:
                        pass
                    last_time = time.time()
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

if __name__ == '__main__':
    main()
