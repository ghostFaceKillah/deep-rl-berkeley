import argparse
import gym
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.contrib.layers as layers

import logz
import policies
import utils

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)


class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)

    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


class NnValueFunction(object):
    """
    Inspired by Daniel Seita's implementation
    https://github.com/DanielTakeshi/rl_algorithms/blob/master/utils/value_functions.py
    """

    def __init__(self, session, ob_dim, n_epochs=20):
        self.obs = tf.placeholder(shape=[None, 2 * ob_dim], name='nn_val_func_ob', dtype=tf.float32)
        self.h1 = layers.fully_connected(
            self.obs,
            num_outputs=50,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=tf.nn.elu
        )
        self.h2 = layers.fully_connected(
            self.obs,
            num_outputs=50,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=tf.nn.elu
        )
        self.y_pred = layers.fully_connected(
            self.h2,
            num_outputs=1,
            weights_initializer=layers.xavier_initializer(uniform=True)
        )
        self.y_pred = tf.reshape(self.y_pred, [-1]) # (?, 1) -> (?,)

        # For the loss function, which is the simple (mean) L2 error
        self.sess      = session
        self.n_epochs  = n_epochs
        self.y_targets = tf.placeholder(shape=[None], name='nn_val_func_target', dtype=tf.float32)
        self.loss      = tf.losses.mean_squared_error(self.y_targets, self.y_pred)
        self.train_op  = tf.train.AdamOptimizer().minimize(self.loss)

    def fit(self, X, y):
        """
        Update the value function based on current batch of observations
        """
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        Xp = self.preproc(X)
        for _ in range(self.n_epochs):
            _, _ = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self.obs: Xp,
                    self.y_targets: y
                }
            )

    def predict(self, X):
        """ Estimate value of given state"""
        # Should we expand state dim to from (n,) to (n, 1)?

        Xp = self.preproc(X)
        return self.sess.run(self.y_pred, feed_dict={self.obs: Xp})

    def preproc(self, X):
        """Add some nonlinearity to the inputs """
        return np.concatenate([X, np.square(X) / 2.0], axis=1)


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def run_vanilla_policy_gradient_experiment(args, vf_params, logdir, env, sess, continuous_control):
    """
    General purpose method to run vanilla policy gradients.
    Works for both continuous and discrete environments.

    Roughly inspired by starter code for this homework and
    https://github.com/DanielTakeshi/rl_algorithms/blob/master/vpg/main.py

    Thanks!

    Params
    ------
    args: arguments for vanilla policy gradient.
    vf_params: dict of params for value function
    logdir: where to store outputs or None if you don't want to store anything
    env: openai gym env
    sess: TF session
    continuous_control: boolean, if true then we do gaussian continuous control
    """

    ob_dim = env.observation_space.shape[0]

    if args.vf_type == 'linear':
        value_function = LinearValueFunction(**vf_params)
    elif args.vf_type == 'nn':
        value_function = NnValueFunction(session=sess, ob_dim=ob_dim)

    if continuous_control:
        ac_dim = env.action_space.shape[0]
        policy_fn = policies.GaussianPolicy(sess, ob_dim, ac_dim)
    else:
        ac_dim = env.action_space.n
        policy_fn = policies.DisceretePolicy(sess, ob_dim, ac_dim)


    sess.__enter__()  # equivalent to with sess, to reduce indentation
    tf.global_variables_initializer().run()
    total_timesteps = 0
    stepsize = args.initial_stepsize

    for i in range(args.n_iter):
        print("\n********** Iteration %i ************" % i)

        # Collect paths until we have enough timesteps.
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (
            len(paths) == 0 and (i % 10 == 0) and args.render)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = policy_fn.sample_action(ob)
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break
            path = {"observation": np.array(obs), "terminated": terminated,
                    "reward": np.array(rewards), "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += utils.pathlength(path)
            if timesteps_this_batch > args.min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Estimate advantage function using baseline vf (these are lists!).
        # return_t: list of sum of discounted rewards (to end of
        # episode), one per time
        # vpred_t: list of value function's predictions of components of
        # return_t
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = utils.discount(rew_t, args.gamma)
            vpred_t = value_function.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update and **re-fit the baseline**.
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        std_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        value_function.fit(ob_no, vtarg_n)

        # Policy update, plus diagnostics stuff. Is there a better way to
        #  handle
        # the continuous vs discrete control cases?
        if continuous_control:
            surr_loss, oldmean_na, oldlogstd_a = policy_fn.update_policy(
                ob_no, ac_n, std_adv_n, stepsize)

            kl, ent = policy_fn.kldiv_and_entropy(
                ob_no, oldmean_na, oldlogstd_a
            )
        else:
            surr_loss, oldlogits_na = policy_fn.update_policy(
                ob_no, ac_n, std_adv_n, stepsize)
            kl, ent = policy_fn.kldiv_and_entropy(ob_no, oldlogits_na)

        # Step size heuristic to ensure that we don't take too large steps.
        if args.use_kl_heuristic:
            if kl > args.desired_kl * 2:
                stepsize /= 1.5
                print('PG stepsize -> %s' % stepsize)
            elif kl < args.desired_kl / 2:
                stepsize *= 1.5
                print('PG stepsize -> %s' % stepsize)
            else:
                print('PG stepsize OK')

        # Log diagnostics
        if i % args.log_every_t_iter == 0:
            logz.log_tabular("EpRewMean", np.mean(
                [path["reward"].sum() for path in paths]))
            logz.log_tabular("EpLenMean", np.mean(
                [utils.pathlength(path) for path in paths]))
            logz.log_tabular("KLOldNew", kl)
            logz.log_tabular("Entropy", ent)
            logz.log_tabular("EVBefore",
                             utils.explained_variance_1d(vpred_n, vtarg_n))
            logz.log_tabular("EVAfter",
                             utils.explained_variance_1d(value_function.predict(ob_no),
                                                         vtarg_n))
            logz.log_tabular("SurrogateLoss", surr_loss)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            # If you're overfitting, EVAfter will be way larger than
            # EVBefore.
            # Note that we fit the value function AFTER using it to
            # compute the
            # advantage function to avoid introducing bias
            logz.dump_tabular()



def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument('--envname', type=str, default='CartPole-v0')
    p.add_argument('--envname', type=str, default='Pendulum-v0')
    # p.add_argument('--render', action='store_true')
    p.add_argument('--render', default=True)
    p.add_argument('--do_not_save', action='store_true')
    p.add_argument('--use_kl_heuristic', default=True)

    p.add_argument('--n_iter', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gamma', type=float, default=0.97)
    p.add_argument('--desired_kl', type=float, default=2e-3)
    p.add_argument('--min_timesteps_per_batch', type=int, default=2500)
    p.add_argument('--initial_stepsize', type=float, default=1e-3)
    p.add_argument('--log_every_t_iter', type=int, default=1)

    p.add_argument('--vf_type', type=str, default='nn')
    p.add_argument('--nnvf_epochs', type=int, default=20)
    p.add_argument('--nnvf_ssize', type=float, default=1e-3)
    args = p.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    # Handle value function type and the log directory (and save the args!).
    assert args.vf_type == 'linear' or args.vf_type == 'nn'
    vf_params = {}
    outstr = 'linearvf-kl' + str(args.desired_kl)
    if args.vf_type == 'nn':
        vf_params = dict(n_epochs=args.nnvf_epochs,
                         stepsize=args.nnvf_ssize)
        outstr = 'nnvf-kl' + str(args.desired_kl)
    outstr += '-seed' + str(args.seed).zfill(2)
    logdir = 'outputs/' + args.envname + '/' + outstr
    if args.do_not_save:
        logdir = None
    logz.configure_output_dir(logdir)
    if logdir is not None:
        with open(logdir + '/args.pkl', 'wb') as f:
            pickle.dump(args, f)
    print("Saving in logdir: {}".format(logdir))

    # Other stuff for seeding and getting things set up.
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.envname)
    continuous = True
    if 'discrete' in str(type(env.action_space)).lower():
        # A bit of a hack, is there a better way to do this?  Another option
        # could be following Jonathan Ho's code and detecting spaces.Box?
        continuous = False
    print("Continuous control? {}".format(continuous))
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)

    run_vanilla_policy_gradient_experiment(args, vf_params, logdir, env, sess, continuous)
