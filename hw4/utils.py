import numpy as np
import scipy.signal
import tensorflow as tf


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


def gauss_log_prob(mu, logstd, x):
    """
    Write me plz
    """
    var_na = tf.exp(2  * logstd)
    gp_na = -tf.square(x - mu) / (2 * var_na) - 0.5 * tf.log(tf.constant(2 * np.pi)) - logstd
    return tf.reduce_sum(gp_na, axis=[1])


def gauss_KL(mu_1, logstd_1, mu_2, logstd_2):
    """
    Returns KL divergence among two multivariate Gaussians, component-wise.
    """
    var_1_na = tf.exp(2.0 * logstd_1)
    var_2_na = tf.exp(2.0 * logstd_2)
    tmp_matrix = 2.*(logstd_2 - logstd_1) + (var_1_na + tf.square(mu_1-mu_2)) / var_2_na - 1
    kl_n = tf.reduce_sum(0.5 * tmp_matrix, axis=[1]) # Don't forget the 1/2 !!

    # Make sure it passes sanity check
    assert_op = tf.Assert(tf.reduce_all(kl_n >= -0.0000001), [kl_n])
    with tf.control_dependencies([assert_op]):
        kl_n = tf.identity(kl_n)

    return kl_n


def pathlength(path):
    return len(path["reward"])


def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
