import cv2
import tensorflow as tf
import numpy as np
from dataset import Dataset

# Just disables the warning, doesn't enable AVX/FMA cf
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not
# -compiled-to-u/50073238
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork:
    def __init__(self, gpu_check=True):
        self.nothing = None
        self.dataset = Dataset()

        if gpu_check:  # Check that GPU is correctly detected
            tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Hyperparameters
        self.n_inputs = 28 * 284
        self.n_hidden_1 = 3005
        self.n_hidden_2 = 1006
        self.n_outputs = 1078
        self.X = tf.placeholder(tf.float32, shape(None, self.n_inputs),
                                name="X")  # input layer
        self.y = tf.placeholder(tf.int64, shape(None),
                                name="y")  # output layer

        # Define network model
        with tf.name_scope("my_net"):
            hidden_1 = tf.layers.dense(self.X, self.n_hidden_1, name="hidden_1", activation=tf.nn.relu)
            hidden_2 = tf.layers.dense(hidden_1, self.n_hidden_2, name="hidden_2", activation=tf.nn.relu)
            logits = tf.layers.dense(hidden_2, self.n_outputs, name="outputs")  # output before final activation function

        # Define loss function
        with tf.name_scope("loss"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

        # Define optimizer
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(0.0001)
            training_op = optimizer.minimize(loss)

        # Define evaluation metric
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)  # Doest the highest logit match with the class?
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, batch_size, n_epochs):
        # Train the network
        with tf.Session as sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(mnist.train.num_examples / batch_size):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={self.X: X_batch, self.y: y_batch})

                    training_acc = accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                    val_acc = accuracy.eval(feed_dict={self.X: mnist.validation_images, self.y: mnist_validation_labels})

    def test(self):
        # Test the network
        with tf.Session as sess:
            saver.restore(sess, "./my_path/model.ckpt")
            Z = logits.eval(feed_dict={self.X: mnist.testing_images})
            y_pred = np.argmax(Z, axis=1)
