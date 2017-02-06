#!/usr/bin/env python

"""
Dataset aggregation.

Steps:
    1) train cloned policy Pi(u_t, o_t) from expert data
    2) run Pi(u_t, o_t) to get data set D_pi = {o_1, ... , o_M}
    3) Ask human to label D_pi with actions
    4) Aggregate the dataset
"""

import ipdb
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
    "Reacher-v1",
    "Ant-v1",
    "HalfCheetah-v1",
    "Hopper-v1",
    "Humanoid-v1",
    "Walker2d-v1"
]


def load_policy_fn(env_name):
    print('Gathering expert data')
    print('loading and building expert policy')
    policy_fname = 'experts/{}.pkl'.format(env_name)
    policy_fn = load_policy.load_policy(policy_fname)
    print('loaded and built')
    return policy_fn


def get_batch_of_full_expert_data(policy_fn, env, render=False):
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
            if render:
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


def test_run_our_model(model, env, rollouts=20, render=False):
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    steps_numbers = []

    for i in tqdm.tqdm(range(rollouts)):
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
            if render:
                env.render()
            if steps >= max_steps:
                break
        steps_numbers.append(steps)
        returns.append(totalr)

    return {'observations': np.array(observations),
            'actions': np.array(actions),
            'returns': np.array(returns),
            'steps': np.array(steps_numbers)}



def run_expert_on_observations(observations, policy_fn):
    with tf.Session():
        tf_util.initialize()

        actions = []

        for obs in observations:
            action = policy_fn(obs[None,:])
            actions.append(action)

    return np.array(actions)


def build_net(data, env):
    mean = np.mean(data['observations'], axis=0)
    std = np.std(data['observations'], axis=0) + 1e-6

    observations_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]

    model = Sequential([
        Lambda(lambda x: (x - mean) / std,
               batch_input_shape=(None, observations_dim)),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(actions_dim)
    ])

    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    return model


def get_training_opts():
    return dict(validation_split=0.1,
                batch_size=256,
                nb_epoch=5,
                verbose=2)


def extract_stats(data):
    mean = data['returns'].mean()
    std = data['returns'].std()
    x = data['steps']
    pct_full_steps =  (x / x.max()).mean()

    return pd.Series({
        'mean reward': mean,
        'std reward': std,
        'pct full rollout': pct_full_steps
    })


def run_dagger(env_name):
    policy_fn = load_policy_fn(env_name)

    env = gym.make(env_name)
    actions_dim = env.action_space.shape[0]

    training_opts = get_training_opts()

    data = get_batch_of_full_expert_data(policy_fn, env, render=False)
    x = data['observations']
    y = data['actions'].reshape(-1, actions_dim)

    model = build_net(data, env)

    stats = {}
    rewards = {}

    for i in range(50):
        # 1) train cloning policy from expert data
        x, y = shuffle(x, y)
        model.fit(x, y, **training_opts)

        # 2) run cloning policy to get new data
        data = test_run_our_model(model, env, render=False)

        new_x = data['observations']
        stats[i] = extract_stats(data)
        rewards[i] = data['returns']

        # 3) ask expert to label D_pi with actions
        new_y = run_expert_on_observations(new_x, policy_fn)

        # 4) Aggregate the dataset
        x = np.append(x, new_x, axis=0)
        y = np.append(y, new_y.reshape(-1, actions_dim), axis=0)


    df = pd.DataFrame(stats).T
    df.index.name = 'iterations'
    df.to_csv('data/{}-DAgger.csv'.format(env_name))
    pickle_name = 'data/{}-DAgger-rewards.p'.format(env_name)
    pickle.dump(rewards, open(pickle_name, 'wb'))


def show_results(env_name):
    from run_imitation_learning import get_epoch_grid_configs

    pickle_name = 'data/{}-DAgger-rewards.p'.format(env_name)
    returns = pickle.load(open(pickle_name, 'rb'))

    their_data = pickle.load(open('data/{}-their.p'.format(env_name), 'rb'))['returns']
    their_data = np.array([their_data for _ in xrange(len(returns.keys()))]).T

    acc = {}
    configs = get_epoch_grid_configs(env_name)
    for config in configs:
        data = pickle.load(open(config['our_data_path'], 'rb'))['returns']
        lr = config['epochs']
        acc[lr] = data

    vanilla_data = pd.DataFrame(acc)

    df = pd.DataFrame(returns)

    sns.tsplot(time=vanilla_data.columns, data=vanilla_data.values, color='purple', linestyle=':')
    sns.tsplot(time=df.columns, data=df.values, color='blue', linestyle='-')
    sns.tsplot(data=their_data, color='red', linestyle='--')

    plt.ylabel("Mean reward")
    plt.xlabel("Number of epochs (vanilla cloning), number of iterations (DAgger)")

    plt.title("{} - Comparison of DAgger and vanilla behavioral cloning policies".format(env_name))

    # dirty hack for fix a legend not showing up..
    import matplotlib.patches as mpatches
    plt.legend(handles=[
        mpatches.Patch(color='purple', label='Vanilla cloning policy'),
        mpatches.Patch(color='blue', label='DAgger policy'),
        mpatches.Patch(color='red', label='expert policy'),
    ], loc='lower right')

    plt.savefig('imgs/dagger-vanilla-comp-{}.png'.format(env_name))
    plt.close()


if __name__ == '__main__':
    for task in TASK_LIST:
        # run_dagger(task)
        show_results(task)
