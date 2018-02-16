from __future__ import division, print_function, absolute_import

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import TSNE
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters

INPUT_SIZE = 784
FC_ENCODER = 512
FC_ENCODER_2 = 256
FC_ENCODER_3 = 128
EMBEDDING_DIM = 100
FC_DECODER = 128
FC_DECODER_2 = 256
FC_DECODER_3 = 512


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


def deep_autoencoder(x_input, keep_prob):
    # Model params
    # Encoder
    W1 = weight_variable([INPUT_SIZE, FC_ENCODER])
    B1 = bias_variable([FC_ENCODER])
    W2 = weight_variable([FC_ENCODER, FC_ENCODER_2])
    B2 = bias_variable([FC_ENCODER_2])
    W3 = weight_variable([FC_ENCODER_2, FC_ENCODER_3])
    B3 = bias_variable([FC_ENCODER_3])
    W4 = weight_variable([FC_ENCODER_3, EMBEDDING_DIM])
    B4 = bias_variable([EMBEDDING_DIM])

    # Decoder
    W1_decoder = weight_variable([EMBEDDING_DIM, FC_DECODER])
    B1_decoder = bias_variable([FC_DECODER])
    W2_decoder = weight_variable([FC_DECODER, FC_DECODER_2])
    B2_decoder = bias_variable([FC_DECODER_2])
    W3_decoder = weight_variable([FC_DECODER_2, FC_DECODER_3])
    B3_decoder = bias_variable([FC_DECODER_3])
    W4_decoder = weight_variable([FC_DECODER_3, INPUT_SIZE])
    B4_decoder = bias_variable([INPUT_SIZE])

    # The model
    x_input = tf.reshape(x_input, [-1, INPUT_SIZE])
    with tf.name_scope('encoder'):
        # Enocder Hidden layer with relu activation #1
        layer_1 = tf.nn.relu(tf.matmul(x_input, W1) + B1)
        fc1_drop = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = tf.nn.relu(tf.matmul(fc1_drop, W2) + B2)
        fc2_drop = tf.nn.dropout(layer_2, keep_prob)
        layer_3 = tf.nn.relu(tf.matmul(fc2_drop, W3) + B3)
        fc3_drop = tf.nn.dropout(layer_3, keep_prob)
        encoded = tf.nn.relu(tf.matmul(fc3_drop, W4) + B4)


    with tf.name_scope('decoder'):
        # Decoder Hidden layer with relu activation #1
        decoder_layer_1 = tf.nn.relu(tf.matmul(encoded, W1_decoder) + B1_decoder)
        decoder_layer_1_drop = tf.nn.dropout(decoder_layer_1, keep_prob)
        decoder_layer_2 = tf.nn.relu(tf.matmul(decoder_layer_1_drop, W2_decoder) + B2_decoder)
        decoder_layer_2_drop = tf.nn.dropout(decoder_layer_2, keep_prob)
        decoder_layer_3 = tf.nn.relu(tf.matmul(decoder_layer_2_drop, W3_decoder) + B3_decoder)
        decoder_layer_3_drop = tf.nn.dropout(decoder_layer_3, keep_prob)
        decoded = tf.nn.relu(tf.matmul(decoder_layer_3_drop, W4_decoder) + B4_decoder)

    return decoded, encoded


# The deep model
def deep_autoencoder_tensorflow(batch_size=64, learning_rate=1e-2, dropout_prob=1., num_steps=10000):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs_path = '/tmp/auto_encoder/{}_dropout_{}_lr_{}_batch_{}'.format(timestamp, dropout_prob, learning_rate,
                                                                             batch_size)
    g = tf.Graph()
    with g.as_default():
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        dummy_x_embedding_input = np.zeros((batch_size, EMBEDDING_DIM))
        keep_prob = tf.placeholder(tf.float32)

        predicted_image, embedding = deep_autoencoder(x, keep_prob)

        loss = tf.reduce_mean(tf.pow(x - predicted_image, 2))

        with tf.name_scope('Optimaizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        print("Start trainig, log_path {}".format(logs_path))
        # Start training
        sess = tf.Session()
        with sess.as_default():
            sess.run(init)
            train_writer = tf.summary.FileWriter(logs_path + r"/train", graph=tf.get_default_graph())
            test_writer = tf.summary.FileWriter(logs_path + r"/test", graph=tf.get_default_graph())

            for i in range(num_steps):
                batch_xs, _ = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                if i % 200 == 0:
                    _, _, train_summary = sess.run([optimizer, loss, merged_summary_op],
                                                   feed_dict={x: batch_xs, keep_prob: dropout_prob})
                    train_writer.add_summary(train_summary, i)
                if i % 500 == 0:
                    test_batch, _ = mnist.train.next_batch(batch_size)
                    cost, test_summary = sess.run([loss, merged_summary_op],
                                                  feed_dict={x: test_batch, keep_prob: 1})
                    test_writer.add_summary(test_summary, i)
                    print("iter{}: cost:{}".format(i, cost))
                else:
                    optimizer.run(
                        feed_dict={x: batch_xs, keep_prob: dropout_prob})

        print("Optimization Finished!")

        #### PLOT EMBEDDING ####
        test_set, _ = mnist.test.next_batch(10000)
        _, test_summary, emb = sess.run([loss, merged_summary_op, embedding],
                                        feed_dict={x: test_set, keep_prob: 1})
        X = tsne(emb)

        plot_with_images(X, test_set, "AUTOENCODER - MNIST", image_num=100)
        plt.savefig("autoencode_mnist")
        plt.show()


        # Encode and decode images from test set and visualize their reconstruction
        n = 4
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))
        for i in range(n):
            # MNIST test set
            batch_x, _ = mnist.test.next_batch(n)
            # Encode and decode the digit image
            reconstruction = sess.run(predicted_image,
                                      feed_dict={x: batch_x, keep_prob: 1.})

            # Display original images
            for j in range(n):
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(n):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    reconstruction[j].reshape([28, 28])

        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.title('Original Images')

        plt.figure(figsize=(n, n))
        plt.title('Reconstructed Images')
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()
    return

def tsne(X, k=2,perplexity=100):
    tsne = TSNE(n_components=k, init='pca', random_state=0, perplexity=perplexity)
    X_transformed = tsne.fit_transform(X)
    # pca = RandomizedPCA(n_components=2)
    # X_transformed = pca.fit_transform(X)

    return X_transformed


def plot_with_images(X, images, title="", image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)
    plt.show()
    return fig


if __name__ == '__main__':
    # test_set, _ = mnist.train.next_batch(10000)
    deep_autoencoder_tensorflow(dropout_prob=0.9)
