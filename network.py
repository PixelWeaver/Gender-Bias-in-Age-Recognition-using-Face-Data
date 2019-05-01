import tensorflow as tf


def network_def(input, mode):
    filters = [96, 256, 384]
    kernel_sizes = [7, 5, 3]
    pool_sizes = [5, 3, 2]
    dropout_rates = [0.2, 0.4, 0.7]
    iterative_layer = input

    # Convolutional blocks
    for filter_num, kernel_size, pool_size, dropout_rate in zip(filters, kernel_sizes, pool_sizes, dropout_rates):
        cv_layer = tf.layers.conv2d(
            inputs=iterative_layer,
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu)

        pool = tf.layers.max_pooling2d(inputs=cv_layer, pool_size=[pool_size, pool_size], strides=2)

        iterative_layer = tf.layers.dropout(
            inputs=pool, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer
    flattened_pool = tf.layers.flatten(iterative_layer)
    dense_layer_common = tf.layers.dense(inputs=flattened_pool, units=1024, activation=tf.nn.relu)
    dropout_layer = tf.layers.dropout(
        inputs=dense_layer_common, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Gender subnetwork
    gender_dense_layer = tf.layers.dense(inputs=dropout_layer, units=1024)
    gender_output = tf.layers.dense(inputs=gender_dense_layer, units=2)

    # Age subnetwork
    age_dense_layer = tf.layers.dense(inputs=dropout_layer, units=1024)
    age_output = tf.layers.dense(inputs=age_dense_layer, units=8)

    return age_output, gender_output



