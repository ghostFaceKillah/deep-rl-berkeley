#!/usr/bin/env python

"""
Code to run imitation learning experiments, based on expert policies provided by the course staff.

It implements two main modes:
- vanilla run of a neural network policy
- grids over some basic parameters, such as learning rate, epoch size and so on ...

Author of included expert policies: Jonathan Ho (hoj@openai.com)
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


def train_net(data, env, config):
    """
    What about learning rate?
    What about epochs?
    """
    from sklearn.utils import shuffle

    from keras.models import Sequential
    from keras.layers import Dense, Lambda
    from keras.optimizers import Adam

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
    x, y = shuffle(data['observations'], data['actions'].reshape(-1, actions_dim))
    model.fit(x, y,
              validation_split=0.1,
              batch_size=256,
              nb_epoch=config['epochs'],
              verbose=2)

    return model


def run_single_experiment(config):
    env = gym.make(config['env_name'])

    if config['use_cached_data_for_training']:
        data = pickle.load(open(config['cached_data_path'], 'rb'))
    else:
        data = gather_expert_data(config, env)

    model = train_net(data, env, config)
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


def analyze_single_experiment_data(config):
    their = pickle.load(open(config['their_data_path'], 'rb'))
    our = pickle.load(open(config['our_data_path'], 'rb'))

    df = pd.DataFrame({
        'expert': one_data_table_stats(their),
        'imitation': one_data_table_stats(our)
    })

    print "Analyzing experiment {}".format(config['env_name'])
    print df


def get_default_config(env_name):
    return {
        'env_name': env_name,
        'expert_policy_file': 'experts/{}.pkl'.format(env_name),
        'envname': env_name,
        'render_them': False,
        'render_us': False,
        'num_rollouts': 30,
        'use_cached_data_for_training': True,
        'cached_data_path': 'data/{}-their.p'.format(env_name),
        'their_data_path': 'data/{}-their.p'.format(env_name),
        'our_data_path': 'data/{}-our.p'.format(env_name),
        # neural net params
        'learning_rate': 0.001,
        'epochs': 30
    }


def get_lr_grid_configs(env_name):
    learning_rate_grid = np.logspace(-8, 0, num=13)

    configs = []
    for lr in learning_rate_grid:
        config = get_default_config(env_name)
        config['use_cached_data_for_training'] = True
        config['our_data_path'] = 'data/{}-our-lr={:.8f}.p'.format(env_name, lr)
        config['learning_rate'] = lr
        configs.append(config)

    return configs


def get_epoch_grid_configs(env_name):
    epoch_grid = [1, 2, 5, 10, 15, 20, 30, 40, 50]

    configs = []
    for epochs in epoch_grid:
        config = get_default_config(env_name)
        config['use_cached_data_for_training'] = True
        config['our_data_path'] = 'data/{}-our-epochs={}.p'.format(env_name, epochs)
        config['epochs'] = epochs
        configs.append(config)

    return configs


def run_all_vanilla_experiments():
    # for task in TASK_LIST:
    #     run_single_experiment(get_default_config(task))

    for task in TASK_LIST:
        analyze_single_experiment_data(get_default_config(task))


def run_lr_grid(task):
    configs = get_lr_grid_configs(task)

    # for config in configs:
    #     run_single_experiment(config)

    acc = {}
    for config in configs:
        data = pickle.load(open(config['our_data_path'], 'rb'))['returns']
        lr = config['learning_rate']
        acc[lr] = data

    data = pd.DataFrame(acc)
    their_data = pickle.load(open(config['their_data_path'], 'rb'))['returns']
    their_data = np.array([their_data for _ in xrange(data.shape[1])]).T

    # s = slice(3, -2)

    ax = sns.tsplot(data=data.values, color='blue', legend='Imitation learning', linestyle='-')

    sns.tsplot(data=their_data, color='red', legend='Expert policy', linestyle='--')

    ax.set_xticklabels(['{:.2e}'.format(x) for x in data.columns])

    plt.title('Reward vs learning rate. Task = {}'.format(task))

    plt.savefig('imgs/{}-lr-reward.png'.format(task))
    plt.close()


def run_epoch_grid(task):
    configs = get_epoch_grid_configs(task)

    for config in configs:
        run_single_experiment(config)

    acc = {}
    for config in configs:
        data = pickle.load(open(config['our_data_path'], 'rb'))['returns']
        lr = config['epochs']
        acc[lr] = data

    data = pd.DataFrame(acc)
    their_data = pickle.load(open(config['their_data_path'], 'rb'))['returns']
    their_data = np.array([their_data for _ in xrange(data.shape[1])]).T

    ax = sns.tsplot(data=data.values, color='blue', legend='Imitation learning', linestyle='-')
    sns.tsplot(data=their_data, color='red', legend='Expert policy', linestyle='--')
    ax.set_xticklabels(data.columns)

    plt.title('Reward vs no of epochs. Task = {}'.format(task))

    plt.savefig('imgs/{}-epoch-reward.png'.format(task))
    plt.close()


if __name__ == '__main__':
    run_all_vanilla_experiments()

    for task in tqdm.tqdm(TASK_LIST):
        run_epoch_grid(task)
        run_lr_grid(task)
