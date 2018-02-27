import pickle
import sys

import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

INPUT_SIZE = 784
FC_ENCODER = 256
EMBEDDING_DIM = 100
FC_DECODER = 256

save_to = "autoencoder_model.pkl"


# Helper functions
def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# init_variables try to load from pickle:
try:
    model = pickle.load(open(save_to, 'rb'))

    W1_decoder = tf.Variable(tf.constant(model[4]), name="G_w1_loaded")
    B1_decoder = tf.Variable(tf.constant(model[5]), name="G_b1_loaded")
    W2_decoder = tf.Variable(tf.constant(model[6]), name="G_w2_loaded")
    B2_decoder = tf.Variable(tf.constant(model[7]), name="G_b2_loaded")
    print("model has been loaded from {}".format(save_to))
except:
    # Model params
    # Decoder
    with tf.variable_scope("G_decoder"):
        W1_decoder = weight_variable([EMBEDDING_DIM, FC_DECODER], name="G_w1")
        B1_decoder = bias_variable([FC_DECODER])
        W2_decoder = weight_variable([FC_DECODER, INPUT_SIZE], name="G_w2")
        B2_decoder = bias_variable([INPUT_SIZE])


def deep_autoencoder(embedding):
    with tf.name_scope('decoder'):
        # Decoder Hidden layer with relu activation #1
        decoder_layer_1 = tf.nn.relu(tf.matmul(embedding, W1_decoder) + B1_decoder)
        decoded = tf.nn.relu(tf.matmul(decoder_layer_1, W2_decoder) + B2_decoder)

    return decoded


def plot(samples, D_loss, G_loss, epoch, total):
    fig = plt.figure(figsize=(10, 5))

    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0.05, hspace=0.05)

    # Plot losses
    ax = plt.subplot(gs[:, 4:])
    ax.plot(D_loss, label="discriminator's loss", color='b')
    ax.plot(G_loss, label="generator's loss", color='r')
    ax.set_xlim([0, total])
    ax.yaxis.tick_right()
    ax.legend()

    # Generate images
    for i, sample in enumerate(samples):

        if i > 4 * 4 - 1:
            break
        ax = plt.subplot(gs[i % 4, int(i / 4)])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.savefig('./output/' + str(epoch + 1) + '.png')
    plt.close()


def Conv2d(input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())

    return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b


def Deconv2d(input, output_dim, batch_size, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())

        input_shape = input.get_shape().as_list()
        output_shape = [batch_size,
                        int(input_shape[1] * strides[0]),
                        int(input_shape[2] * strides[1]),
                        output_dim]

        deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
                                        strides=[1, strides[0], strides[1], 1])

        return deconv + b


def Dense(input, output_dim, stddev=0.02, name='dense'):
    with tf.variable_scope(name):
        shape = input.get_shape()
        W = tf.get_variable('DenseW', [shape[1], output_dim], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('Denseb', [output_dim],
                            initializer=tf.zeros_initializer())

        return tf.matmul(input, W) + b


def BatchNormalization(input, name='bn'):
    with tf.variable_scope(name):

        output_dim = input.get_shape()[-1]
        beta = tf.get_variable('BnBeta', [output_dim],
                               initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [output_dim],
                                initializer=tf.ones_initializer())

        if len(input.get_shape()) == 2:
            mean, var = tf.nn.moments(input, [0])
        else:
            mean, var = tf.nn.moments(input, [0, 1, 2])
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)


def LeakyReLU(input, leak=0.2, name='lrelu'):
    return tf.maximum(input, leak * input)

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

BATCH_SIZE = 64
EPOCHS = 40


def Discriminator(X, reuse=False, name='d'):
    with tf.variable_scope(name, reuse=reuse):

        if len(X.get_shape()) > 2:
            # X: -1, 28, 28, 1
            D_conv1 = Conv2d(X, output_dim=64, name='D_conv1')
        else:
            D_reshaped = tf.reshape(X, [-1, 28, 28, 1])
            D_conv1 = Conv2d(D_reshaped, output_dim=64, name='D_conv1')
        D_h1 = LeakyReLU(D_conv1)  # [-1, 28, 28, 64]
        D_conv2 = Conv2d(D_h1, output_dim=128, name='D_conv2')
        D_h2 = LeakyReLU(D_conv2)  # [-1, 28, 28, 128]
        D_r2 = tf.reshape(D_h2, [-1, 256])
        D_h3 = LeakyReLU(D_r2)  # [-1, 256]
        D_h4 = tf.nn.dropout(D_h3, 0.5)
        D_h5 = Dense(D_h4, output_dim=1, name='D_h5')  # [-1, 1]
        return tf.nn.sigmoid(D_h5)


X = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])
# G = Generator(z, 'G')
G_from_decoder = deep_autoencoder(z)
D_real = Discriminator(X, False, 'D')
D_fake = Discriminator(G_from_decoder, True, 'D')

D_loss = -tf.reduce_mean(tf.log(D_real) - tf.log(D_fake))  # Train to judge if the data is real correctly
G_loss = -tf.reduce_mean(tf.log(D_fake))  # Train to pass the discriminator as real data

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G')]
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), d_params)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), g_params)


D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.1).minimize(D_loss, var_list=d_params)
G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.3).minimize(G_loss, var_list=g_params)



# loss_d_real = binary_cross_entropy(tf.ones_like(D_real), D_real)
# loss_d_fake = binary_cross_entropy(tf.zeros_like(D_fake), D_fake)
# G_loss = tf.reduce_mean(binary_cross_entropy(tf.ones_like(D_fake), D_fake))
# D_loss = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss + d_reg, var_list=d_params)
#     G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(G_loss + g_reg, var_list=g_params)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    D_loss_vals = []
    G_loss_vals = []

    iteration = int(mnist.train.images.shape[0] / BATCH_SIZE)
    for e in range(EPOCHS):

        for i in range(iteration):
            x, _ = mnist.train.next_batch(BATCH_SIZE)
            rand = np.random.uniform(0., 1., size=[BATCH_SIZE, EMBEDDING_DIM])
            _, D_loss_curr = sess.run([D_solver, D_loss], {X: x, z: rand})
            rand = np.random.uniform(0., 1., size=[BATCH_SIZE, EMBEDDING_DIM])
            _, G_loss_curr = sess.run([G_solver, G_loss], {z: rand})

            D_loss_vals.append(D_loss_curr)
            G_loss_vals.append(G_loss_curr)

            sys.stdout.write("\r%d / %d: %f, %f" % (i, iteration, D_loss_curr, G_loss_curr))
            sys.stdout.flush()

        data = sess.run(G_from_decoder, {z: rand})
        plot(data, D_loss_vals, G_loss_vals, e, EPOCHS * iteration)
