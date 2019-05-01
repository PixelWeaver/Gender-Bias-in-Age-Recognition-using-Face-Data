import tensorflow as tf
from estimator import model_fn, serving_fn
from dataset import Dataset

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    dataset = Dataset()

    model_dir = "model"
    config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=1500)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: my_input_fn(dataset.train_record_path),
                                        max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: my_input_fn(dataset.val_record_path))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=config, params={
            'learning_rate': 0.0001
        })

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    estimator.export_savedmodel(export_dir_base='{}/serving'.format(model_dir),
                                serving_input_receiver_fn=serving_fn, as_text=True)


def parser(record):
    features = {
        'feats': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    parsed = tf.parse_single_example(record, features)
    feats = tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.float64))
    label = tf.cast(parsed['label'], tf.int32)

    return {'feats': feats}, label


def my_input_fn(tfrecords_path):
    ds = (
        tf.data.TFRecordDataset(tfrecords_path).map(parser).batch(1024)
    )

    iterator = ds.make_one_shot_iterator()

    batch_feats, batch_labels = iterator.get_next()

    return batch_feats, batch_labels
