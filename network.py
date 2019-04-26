 import tensorflow as tf
from dataset import Dataset

# Just disables the warning, doesn't enable AVX/FMA cf
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not
# -compiled-to-u/50073238
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork:
    def __init__(self, gpu_check=True):
        self.dataset = Dataset()

        if gpu_check:  # Check that GPU is correctly detected
            tf.Session(config=tf.ConfigProto(log_device_placement=True))

    def cnn_model_fn(self, mode):
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        # Convolutional Layer #1 + Pooling Layer
        conv_1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=96,
            kernel_size=[7, 7],
            name="conv_1",
            padding="same",
            activation=tf.nn.relu)

        pool_1 = tf.layers.max_pooling2d(
            inputs=conv_1,
            pool_size=[5, 5],
            strides=5)

        # Convolutional Layer #2 + Pooling Layer
        conv_2 = tf.layers.conv2d(
            inputs=pool_1,
            filters=256,
            kernel_size=[5, 5],
            name="conv_2",
            padding="same",
            activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(
            inputs=conv_2,
            pool_size=[3, 3],
            strides=3)

        # Convolutional Layer #3 + Pooling Layer
        conv_3 = tf.layers.conv2d(
            inputs=pool_2,
            filters=384,
            kernel_size=[3, 3],
            name="conv_3",
            padding="same",
            activation=tf.nn.relu)
        pool_3 = tf.layers.max_pooling2d(
            inputs=conv_3,
            pool_size=[2, 2],
            strides=2)

        # Dense Layer #1
        pool2_flat = tf.reshape(pool_3, [-1, 7 * 7 * 64])
        dense_1 = tf.layers.dense(
            inputs=pool2_flat,
            units=512,
            activation=tf.nn.relu)
        dropout_1 = tf.layers.dropout(
            inputs=dense_1,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN)

        # Dense Layer #2
        dense_2 = tf.layers.dense(
            inputs=dropout_1,
            units=512,
            activation=tf.nn.relu)
        dropout_2 = tf.layers.dropout(
            inputs=dense_2,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN)


        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
