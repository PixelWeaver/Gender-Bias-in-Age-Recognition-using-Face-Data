import tensorflow as tf


def parser(record):
    features = {
        'feats': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    parsed = tf.parse_single_example(record, features)
    feats = tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.float64))
    label = tf.cast(parsed['label'], tf.int32)

    return {'feats': feats}, label


def input_fn(ds):
    iterator = ds.make_one_shot_iterator()

    batch_feats, batch_labels = iterator.get_next()

    return batch_feats, batch_labels
