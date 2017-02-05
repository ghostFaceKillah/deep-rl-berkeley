#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v1 --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import gym
import load_policy
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import tf_util
import tqdm


TASK_LIST = [
    "Ant-v1",
    "HalfCheetah-v1",
    "Hopper-v1",
    "Humanoid-v1",
    "Reacher-v1",
    "Walker2d-v1"
]


def gather_expert_data(config, env):
    print('Gathering expert data')
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(config['expert_policy_file'])
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

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

    pickle.dump(expert_data, open(config['their_data_path'], 'wb'))
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


def run_single_experiment(env_name):
    config = get_default_config(env_name)

    env = gym.make(env_name)

    if config['use_cached_data_for_training']:
        data = pickle.load(open(config['cached_data_path'], 'rb'))
    else:
        data = gather_expert_data(config, env)

    model = train_net(data, env)
    test_run_our_model(model, config, env)


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


def analyze_single_experiment_data(env_name):
    config = get_default_config(env_name)

    their = pickle.load(open(config['their_data_path'], 'rb'))
    our = pickle.load(open(config['our_data_path'], 'rb'))

    df = pd.DataFrame({
        'expert': one_data_table_stats(their),
        'imitation': one_data_table_stats(our)
    })

    print "Analyzing experiment {}".format(env_name)
    print df


def get_default_config(env_name):
    return {
        'expert_policy_file': 'experts/{}.pkl'.format(env_name),
        'envname': env_name,
        'render_them': False,
        'render_us': False,
        'num_rollouts': 30,
        'use_cached_data_for_training': False,
        'cached_data_path': 'data/{}-their.p'.format(env_name),
        'their_data_path': 'data/{}-their.p'.format(env_name),
        'our_data_path': 'data/{}-our.p'.format(env_name),
    }


def run_all_experiments():
    for task in TASK_LIST:
        run_single_experiment(task)

    for task in TASK_LIST:
        analyze_single_experiment_data(task)

if __name__ == '__main__':
    pass
