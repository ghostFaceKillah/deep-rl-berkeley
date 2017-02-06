#!/usr/bin/env python

"""
Dataset aggregation.

Steps:
    1) train cloned policy Pi(u_t, o_t) from expert data
    2) run Pi(u_t, o_t) to get data set D_pi = {o_1, ... , o_M}
    3) Ask human to label D_pi with actions
    4) Aggregate the dataset
"""

import gym
import load_policy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import tensorflow as tf
import tf_util
import tqdm

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import Adam


TASK_LIST = [
    "Ant-v1",
    "HalfCheetah-v1",
    "Hopper-v1",
    "Humanoid-v1",
    "Reacher-v1",
    "Walker2d-v1"
]


def load_policy(config):
    print('Gathering expert data')
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(config['expert_policy_file'])
    print('loaded and built')
    return policy_fn


def get_batch_of_full_expert_data(config, policy_fn, env):
    with tf.Session():
        tf_util.initialize()

        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        steps_numbers = []

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
            if config['render_them']:
                env.render()
            if steps >= max_steps:
                break
        steps_numbers.append(steps)
        returns.append(totalr)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns),
                       'steps': np.array(steps_numbers)}

    return expert_data


def test_run_our_model(model, config, env):
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    steps_numbers = []

    for i in tqdm.tqdm(range(config['num_rollouts'])):
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
            if config['render_us']:
                env.render()
            if steps >= max_steps:
                break
        steps_numbers.append(steps)
        returns.append(totalr)

    our_net_data = {'observations': np.array(observations),
                    'actions': np.array(actions),
                    'returns': np.array(returns),
                    'steps': np.array(steps_numbers)}

    pickle.dump(our_net_data, open(config['our_data_path'], 'wb'))
    return our_net_data


def run_expert_on_observations(config, observations, policy_fn):
    with tf.Session():
        tf_util.initialize()

        actions = []

        for obs in observations:
            action = policy_fn(obs[None,:])
            actions.append(action)

    return actions


def build_net(data, env, config):
    mean, std = np.mean(data['observations'], axis=0), np.std(data['observations'], axis=0) + 1e-6

    observations_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]

    model = Sequential([
        Lambda(lambda x: (x - mean) / std, batch_input_shape=(None, observations_dim)),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(actions_dim)
    ])

    opt = Adam(lr=config['learning_rate'])
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model


def run_dagger():
    pass





    model.fit(x, y,
              validation_split=0.1,
              batch_size=256,
              nb_epoch=config['epochs'],
              verbose=2)




def one_data_table_stats(data):
    mean = data['returns'].mean()
    std = data['returns'].std()
    x = data['steps']
    pct_full_steps =  (x / x.max()).mean()

    return pd.Series({
        'mean reward': mean,
        'std reward': std,
        'pct full rollout': pct_full_steps
    })


def get_default_config(env_name):
    return {
        'env_name': env_name,
        'expert_policy_file': 'experts/{}.pkl'.format(env_name),
        'envname': env_name,
        'render_them': False,
        'render_us': False,
        'num_rollouts': 5,
        'use_cached_data_for_training': True,
        'cached_data_path': 'data/{}-their.p'.format(env_name),
        'their_data_path': 'data/{}-their.p'.format(env_name),
        'our_data_path': 'data/{}-our.p'.format(env_name),
        # neural net params
        'learning_rate': 0.001,
        'epochs': 30
    }


if __name__ == '__main__':
