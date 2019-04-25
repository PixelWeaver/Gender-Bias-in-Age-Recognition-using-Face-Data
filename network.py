import cv2
import tensorflow as tf
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA cf
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not
# -compiled-to-u/50073238
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Data has to be preprocessed first !!
def load_data():
    image_index = []
    with open('db/preprocessed/image.index', 'r') as f_handle:
        formatted_image_index = f_handle.readlines()
        for formatted_image_input in formatted_image_index:
            image_index.append(formatted_image_input[:-1])

    age_index = []
    with open('db/preprocessed/age.index', 'r') as f_handle:
        formatted_age_index = f_handle.readlines()
        for formatted_age_input in formatted_age_index:
            values = formatted_age_input[:-1]
            values = values.split(" ")

            age_index.append((float(values[0]), float(values[1])))

    gender_index = []
    with open('db/preprocessed/gender.index', 'r') as f_handle:
        formatted_gender_index = f_handle.readlines()
        for formatted_gender_input in formatted_gender_index:
            gender_index.append(int(formatted_gender_input[:-1]))

    train_count = 500
    test_count = 50

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(0, train_count):
        x_train.append(cv2.imread(image_index[i]))
        y_train.append(age_index[i])

    for i in range(train_count, test_count):
        x_test.append(cv2.imread(image_index[i]))
        y_test.append(age_index[i])

    return (x_train, y_train), (x_test, y_test)


# Check that GPU is correctly detected
tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Hyperparameters
n_inputs = 28*284
n_hidden_1 = 3005
n_hidden_2 = 1006
n_outputs = 1078
X = tf.placeholder(tf.float32, shape(None, n_inputs),
                   name="X")  # input layer
y = tf.placeholder(tf.int64, shape(None),
                   name="y")  # output layer

# Define network model
with tf.name_scope("my_net"):
    hidden_1 = tf.layers.dense(X, n_hidden_1, name="hidden_1", activation=tf.nn.relu)
    hidden_2 = tf.layers.dense(hidden_1, n_hidden_2, name="hidden_2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden_2, n_outputs, name="outputs")  # output before final activation function

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

# Training constants
batch_size = 100
n_epochs = 10

# Train the network
with tf.Session as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples / batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            training_acc = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            val_acc = accuracy.eval(feed_dict={X: mnist.validation_images, y: mnist_validation_labels})

# Test the network
with tf.Session as sess:
    saver.restore(sess, "./my_path/model.ckpt")
    Z = logits.eval(feed_dict={X: mnist.testing_images})
    y_pred = np.argmax(Z, axis=1)
