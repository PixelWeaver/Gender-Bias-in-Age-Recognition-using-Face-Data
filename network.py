import tensorflow as tf


def network(features, mode):
    """
    Network creation function

    :param features:
    :param mode:
    :return:
    """
    filters = [96, 256, 384]
    kernel_sizes = [7, 5, 3]
    pool_sizes = [5, 3, 2]
    dropout_rates = [0.2, 0.4, 0.7]
    conv_layer = None

    for filter_num, kernel_size, pool_size, dropout_rate in zip(filters, kernel_sizes, pool_sizes, dropout_rates):
        conv_layer = conv_block(features, mode, filters=filter_num, dropout=dropout_rate, kernel_size=kernel_size,
                                pool_size=pool_size)

    # Dense Layer
    pool4_flat = tf.layers.flatten(conv_layer)
    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Age Head
    age_dense = tf.layers.dense(inputs=dropout, units=1024)
    age_logits = tf.layers.dense(inputs=age_dense, units=8)

    # Gender head
    gender_dense = tf.layers.dense(inputs=dropout, units=1024)
    gender_logits = tf.layers.dense(inputs=gender_dense, units=2)

    return age_logits, gender_logits


def conv_block(input_layer, mode, filters=64, kernel_size=3, pool_size=2, dropout=0.0):
    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=[kernel_size, kernel_size],
        padding="same",
        activation=tf.nn.relu)

    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[pool_size, pool_size], strides=2)

    dropout_layer = tf.layers.dropout(
        inputs=pool, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN)

    return dropout_layer
