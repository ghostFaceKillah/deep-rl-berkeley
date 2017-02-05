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
import load_policy
import argparse


def gather_expert_data(args, env):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

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
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns)}

    pickle.dump(expert_data, open('expert_data.p', 'wb'))
    return expert_data


def run_our_model(model, args, env):
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
            action = model.predict(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # if args.render:
            env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    our_net_data = {'observations': np.array(observations),
                    'actions': np.array(actions),
                    'returns': np.array(returns)}

    return our_net_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    ENV_NAME = 'Humanoid-v1'

    # args = {
    #     'expert_policy_file', 'experts/{}'.format(ENV_NAME)
    #    parser.add_argument('envname', type=str)
    # }
    return args


def train_net(data, env):
    """
    What about learning rate?
    What about epochs?
    """
    from sklearn.utils import shuffle

    from keras.models import Sequential
    from keras.layers import Dense, Lambda

    mean, std = np.mean(data['observations'], axis=0), np.std(data['observations'], axis=0) + 1e-6

    observations_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]

    model = Sequential([
        Lambda(lambda x: (x - mean) / std, batch_input_shape=(None, observations_dim)),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(actions_dim)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    x, y = shuffle(data['observations'], data['actions'].reshape(-1, actions_dim))
    model.fit(x, y,
              validation_split=0.1,
              batch_size=256,
              nb_epoch=30,
              verbose=2)

    return model


def main():
    args = parse_args()
    env = gym.make(args.envname)
    # data = gather_expert_data(args, env)
    data = pickle.load(open('expert_data.p', 'rb'))

    model = train_net(data, env)
    our_data = run_our_model(model, args, env)


if __name__ == '__main__':
    main()
