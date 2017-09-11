import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import utils


class DisceretePolicy(object):
    """
    Also known as Gibbs policy, due to Gibbs sampling I guess?

    A policy in which action is sample based on categorical random variable.
    """

    def __init__(self, sess, ob_dim, ac_dim):
        self.sess = sess

        # Placeholders for inputs
        self.ob_no = tf.placeholder(shape=[None, ob_dim], name='observation', dtype=tf.float32)
        # Actions are ints (categoricals)
        self.ac_n = tf.placeholder(shape=[None], name='action', dtype=tf.int32)

        # Advantage = actual discounted reward for this run - predicted reward
        self.adv_n = tf.placeholder(shape=[None], name='advantage', dtype=tf.float32)
        self.oldlogits_na = tf.placeholder(shape=[None, ac_dim], name='oldlogits', dtype=tf.float32)

        # Form the policy network and the log probabilities
        self.h1 = layers.fully_connected(
            self.ob_no,
            num_outputs=50,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=tf.nn.elu
        )
        self.logits_na = layers.fully_connected(
            self.h1,
            num_outputs=ac_dim,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=None
        )
        self.logp_na = tf.nn.log_softmax(self.logits_na)

        # Log probabilities of the actions in the minibatch
        self.nbatch = tf.shape(self.ob_no)[0]
        self.logprob_n = utils.fancy_slice_2d(self.logp_na, tf.range(self.nbatch), self.ac_n)
        self.sampled_ac = utils.categorical_sample_logits(self.logits_na)[0]

        # Policy gradients loss function and training step
        self.surr_loss = - tf.reduce_mean(self.logprob_n * self.adv_n)
        self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr_loss)

        # For diagnostic purposes
        # These are computed as averages across individual KL / entropy w.r.t. each minibatch state
        self.oldlogp_na = tf.nn.log_softmax(self.oldlogits_na)
        self.oldp_na = tf.exp(self.oldlogp_na)
        self.p_na = tf.exp(self.logp_na)
        self.kl_n = tf.reduce_mean(self.oldp_na * (self.oldlogp_na - self.logp_na), axis=1)

        # What does this do actually? I don't get it
        self.assert_op = tf.Assert(tf.reduce_all(self.kl_n >= -1e4), [self.kl_n])

        with tf.control_dependencies([self.assert_op]):
            self.kl_n = tf.identity(self.kl_n)
        self.kl = tf.reduce_mean(self.kl_n)
        self.ent = tf.reduce_mean(tf.reduce_sum(-self.p_na * self.logp_na, axis=1))

    def sample_action(self, ob):
        return self.sess.run(self.sampled_ac, feed_dict={self.ob_no: ob[None]})

    def update_policy(self, ob_no, ac_n, std_adv_n, stepsize):
        """
        Based on current batch of operations
        """
        feed_dict = {
            self.ob_no: ob_no,
            self.ac_n: ac_n,
            self.adv_n: std_adv_n,   # ???
            self.stepsize: stepsize
        }

        _, surr_loss, oldlogits_na = self.sess.run(
            [self.update_op, self.surr_loss, self.logits_na],
            feed_dict=feed_dict
        )

        return surr_loss, oldlogits_na

    def kldiv_and_entropy(self, ob_no, oldlogits_na):
        """

        """
        return self.sess.run(
            [self.kl, self.ent],
            feed_dict={
                self.ob_no: ob_no,
                self.oldlogits_na: oldlogits_na
            }
        )


class GaussianPolicy(object):
    """
    A policy for continious action spaces,
     where action is sampled from gaussian distribution.
    The parameters learned are the parameters of the gaussian,
    which are conditional on observation.
    """

    def __init__(self, sess, ob_dim, ac_dim):
        self.sess = sess

        # inputs
        self.ob_no = tf.placeholder(shape=[None, ob_dim], name='observations', dtype=tf.float32)

        # Note that actions are
        self.ac_na = tf.placeholder(shape=[None, ac_dim], name="action", dtype=tf.float32)
        self.adv_n = tf.placeholder(shape=[None], name="advantage", dtype=tf.float32)
        self.n = tf.shape(self.ob_no)[0]

        # log of std of our action is a parameter
        self.logstd_a = tf.get_variable("logstd", [ac_dim], initializer=tf.zeros_initializer())
        self.oldlogstd_a = tf.get_variable(name='oldlogstd', shape=[ac_dim], dtype=tf.float32)

        # spread dat stuff
        self.logstd_na = tf.ones(shape=(self.n, ac_dim), dtype=tf.float32) * self.logstd_a
        self.oldlogstd_na = tf.ones(shape=(self.n, ac_dim), dtype=tf.float32) * self.oldlogstd_a

        # Policy network predict the mean of the Gaussian
        self.hidden_1 = layers.fully_connected(
            self.ob_no,
            num_outputs=32,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=tf.nn.elu
        )
        self.hidden_2 = layers.fully_connected(
            self.hidden_1,
            num_outputs=32,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=tf.nn.elu
        )
        self.mean_na = layers.fully_connected(
            self.hidden_2,
            num_outputs=ac_dim,
            weights_initializer=layers.xavier_initializer(uniform=True),
            activation_fn=None
        )
        self.oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)

        # Diagonal Gaussian distribution for sampling actions and log probabilities
        self.logprob_n = utils.gauss_log_prob(mu=self.mean_na, logstd=self.logstd_a, x=self.ac_na)
        self.sampled_ac = (tf.random_normal(tf.shape(self.mean_na)) * tf.exp(self.logstd_a) + self.mean_na)[0]

        # Loss function to differentate
        self.surr_loss = - tf.reduce_mean(self.logprob_n * self.adv_n)
        self.stepsize = tf.placeholder(shape=[], dtype=tf.float32)
        self.training_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr_loss)

        # KL divergence and entropy
        self.kl = tf.reduce_mean(utils.gauss_KL(self.mean_na, self.logstd_na, self.oldmean_na, self.oldlogstd_na))
        self.ent = 0.5 * ac_dim * tf.log(2.0 * np.pi * np.e) + 0.5 * tf.reduce_sum(self.logstd_a)

    def sample_action(self, ob):
        return self.sess.run(self.sampled_ac, feed_dict={self.ob_no: ob[None]})

    def update_policy(self, ob_no, ac_n, std_adv_n, stepsize):
        """
        Update policy params based on the new batch of info
        """

        feed = {
            self.ob_no: ob_no,
            self.ac_na: ac_n,
            self.adv_n: std_adv_n,
            self.stepsize: stepsize
        }
        _, surr_loss, oldmean_na, oldlogstd_a = self.sess.run(
            [self.training_op, self.surr_loss, self.mean_na, self.logstd_a],
            feed_dict=feed
        )
        return surr_loss, oldmean_na, oldlogstd_a

    def kldiv_and_entropy(self, ob_no, oldmean_na, oldlogstd_a):
        feed = {
            self.ob_no: ob_no,
            self.oldmean_na: oldmean_na,
            self.oldlogstd_a: oldlogstd_a
        }

        return self.sess.run([self.kl, self.ent], feed_dict=feed)


















