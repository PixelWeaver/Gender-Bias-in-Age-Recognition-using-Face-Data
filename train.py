import tensorflow as tf
from estimator import model_fn, serving_fn
from dataset import Dataset
from input import input_fn

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    dataset = Dataset()
    train_dataset, test_dataset, val_dataset = dataset.get_dataset()

    model_dir = "model"
    config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=1500)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_dataset),
                                        max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(val_dataset))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=config, params={
            'learning_rate': 0.0001
        })

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    estimator.export_savedmodel(export_dir_base='{}/serving'.format(model_dir),
                                serving_input_receiver_fn=serving_fn, as_text=True)
