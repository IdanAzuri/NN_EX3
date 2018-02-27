import pickle

import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/dcgan_autoencoder/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

INPUT_SIZE = 784
FC_ENCODER = 256
EMBEDDING_DIM = 100
FC_DECODER = 256

save_to = "autoencoder_model.pkl"
# Helper functions
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.x_dim = 784
        self.name = 'mnist/dcgan_autoencoder/g_net'

    def __call__(self, z):

        with tf.variable_scope(self.name) as vs:
            try:
                model = pickle.load(open(save_to, 'rb'))

                W1_decoder = tf.Variable(tf.constant(model[4]))
                B1_decoder = tf.Variable(tf.constant(model[5]))
                W2_decoder = tf.Variable(tf.constant(model[6]))
                B2_decoder = tf.Variable(tf.constant(model[7]))
                print("model has been loaded from {}".format(save_to))
            except:
            # Decoder
                W1_decoder = weight_variable([EMBEDDING_DIM, FC_DECODER])
                B1_decoder = bias_variable([FC_DECODER])
                W2_decoder = weight_variable([FC_DECODER, INPUT_SIZE])
                B2_decoder = bias_variable([INPUT_SIZE])

            decoder_layer_1 = tf.nn.relu(tf.matmul(z, W1_decoder) + B1_decoder)
            decoded = tf.nn.relu(tf.matmul(decoder_layer_1, W2_decoder) + B2_decoder)
            return decoded

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]