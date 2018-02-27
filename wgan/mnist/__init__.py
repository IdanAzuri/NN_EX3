import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def __call__(self, batch_size):
        return mnist.train.next_batch(batch_size)[0]

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])


class NoiseSamplerMultivariable(object):

    def _multivariate_dist(self, batch_size, embedding_dim=100, n_distributions=10):
        current_dist_states_indices = np.random.randint(0, n_distributions - 1, batch_size)
        mean_vec = np.arange(0, embedding_dim, n_distributions)
        cov_mat = np.eye(n_distributions, n_distributions) * np.random.randint(1, 5,
                                                                               n_distributions)  # this is diagonal beacuse we want iid

        result_vec = np.zeros((batch_size, embedding_dim))
        for i in range(batch_size):
            result_vec[i] = np.random.multivariate_normal(mean_vec, cov_mat, size=batch_size * embedding_dim).reshape(embedding_dim,
                                                                                                                      n_distributions,
                                                                                                                      batch_size)[:,
                            current_dist_states_indices[i], i]
            return result_vec

    def __call__(self, batch_size, z_dim):
        return self._multivariate_dist(batch_size, z_dim)

##TF version - NOT WORKING!
# class NoiseSamplerMultivariable(object):
#
#     def _multivariate_dist(self, batch_size, embedding_dim=100, n_distributions=10):
#         # current_dist_states_indices = np.random.randint(0, n_distributions - 1, batch_size)
#         # First sample from [range_start, range_end); maxval is exclusive
#         current_dist_states_indices = tf.random_uniform(dtype=tf.float32, minval=0, maxval=n_distributions - 1, shape=[batch_size])
#         # Increment for values >= the removed element
#         # mean_vec = np.arange(0, embedding_dim, n_distributions)
#         mean_vec=tf.range(0, embedding_dim, delta=3, dtype=tf.float32, name='range')
#
#         # cov_mat = np.eye(n_distributions, n_distributions) * np.random.randint(1, 5, n_distributions)  # this is diagonal beacuse we want iid
#         cov_mat = tf.eye(n_distributions, n_distributions) * tf.random_uniform(dtype=tf.float32, minval=1, maxval=5,
#                                                                                shape=[n_distributions])
#         # result_vec = np.zeros((batch_size, embedding_dim))
#         # result_vec=tf.zeros((batch_size, embedding_dim))
#         result_vec=tf.contrib.distributions.Normal(loc=mean_vec,scale=cov_mat)
#         # result_vec[i] = np.random.multivariate_normal(mean_vec, cov_mat, size=batch_size * embedding_dim).reshape(embedding_dim,
#         #                                                                                                           n_distributions,
#         #                                                                                                           batch_size)[:,
#         #                 current_dist_states_indices[i], i]
#         return result_vec
#     def __call__(self, batch_size, z_dim):
#         return self._multivariate_dist(batch_size, z_dim)

save_to = "autoencoder_model.pkl"

INPUT_SIZE = 784
FC_ENCODER = 256
EMBEDDING_DIM = 100
FC_DECODER = 256

# Helper functions
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv2d_transpose(x):
    '''upsamples the compressed image'''
    return tf.nn.conv2d_transpose(x, filter=32, padding='same', strides=2, name='upsample')


try:
    model = np.pickle.load(open(save_to, 'rb'))
    W1 = tf.Variable(tf.constant(model[0]))
    B1 = tf.Variable(tf.constant(model[1]))
    W2 = tf.Variable(tf.constant(model[2]))
    B2 = tf.Variable(tf.constant(model[3]))
    W1_decoder = tf.Variable(tf.constant(model[4]))
    B1_decoder = tf.Variable(tf.constant(model[5]))
    W2_decoder = tf.Variable(tf.constant(model[6]))
    B2_decoder = tf.Variable(tf.constant(model[7]))
    print ("model has been loaded from {}".format(save_to))
except:
    # Model params
    # Encoder
    W1 = weight_variable([INPUT_SIZE, FC_ENCODER])
    B1 = bias_variable([FC_ENCODER])
    W2 = weight_variable([FC_ENCODER, EMBEDDING_DIM])
    B2 = bias_variable([EMBEDDING_DIM])

    # Decoder
    W1_decoder = weight_variable([EMBEDDING_DIM, FC_DECODER])
    B1_decoder = bias_variable([FC_DECODER])
    W2_decoder = weight_variable([FC_DECODER, INPUT_SIZE])
    B2_decoder = bias_variable([INPUT_SIZE])

def deep_autoencoder(x_input, keep_prob):
    # The model
    x_input = tf.reshape(x_input, [-1, 784])
    with tf.name_scope('encoder'):
        # Enocder Hidden layer with relu activation #1
        layer_1 = tf.nn.relu(tf.matmul(x_input, W1) + B1)
        fc1_drop = tf.nn.dropout(layer_1, keep_prob)
        encoded = tf.nn.relu(tf.matmul(fc1_drop, W2) + B2)

    with tf.name_scope('decoder'):
        # Decoder Hidden layer with relu activation #1
        decoder_layer_1 = tf.nn.relu(tf.matmul(encoded, W1_decoder) + B1_decoder)
        decoder_layer_1_drop = tf.nn.dropout(decoder_layer_1, keep_prob)
        decoded = tf.nn.relu(tf.matmul(decoder_layer_1_drop, W2_decoder) + B2_decoder)

    return decoded, encoded

class AuroEncoderSampler(object):
    def __call__(self,X_placeholder,sess,x_batch,keep_prob=0.9):
        keep_prob = tf.placeholder(tf.float32)
        _, embedding = deep_autoencoder(X_placeholder,keep_prob)
        embedding_out = sess.run(embedding, feed_dict={X_placeholder: x_batch, keep_prob: .9})
        return embedding_out
