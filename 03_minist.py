import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.examples.tutorials.mnist import input_data
import time
import examples.utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('./data/kaggle_mnist/train.csv', nrows=10)
df.describe()
ds = df.as_matrix()

ds[0, :].shape
ds[0, 1:].shape
ds[0, 0].shape

ds[:, 1:].shape

# dataset = tf.data.Dataset.from_tensors((ds[:, 1:], tf.one_hot(ds[:, 0], 10)))
# dataset = dataset.batch(1)
# iterator = dataset.make_initializable_iterator()
# X, Y = iterator.get_next()
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     sess.run(iterator.get_next())

# dataset = tf.data.Dataset.from_sparse_tensor_slices()

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 50

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)

X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.int32, [batch_size, 10], name='label')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

logits = tf.matmul(X, w) + b 
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder',
                               tf.get_default_graph())

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)

    for i in range(n_epochs):
        total_loss = 0
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], {
                X: X_batch,
                Y: Y_batch
            })
            total_loss += loss_batch
            print('Average loss epoch {0}: {1}'.format(i,
                                                       total_loss / n_batches))
        print('Total time: {0} seconds'.format(time.time() - start_time))
        # test the model
        n_batches = int(mnist.test.num_examples / batch_size)
        total_correct_preds = 0

        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(batch_size)
            accuracy_batch = sess.run(accuracy, {X: X_batch, Y: Y_batch})
            total_correct_preds += accuracy_batch

        print('Accuracy {0}'.format(
            total_correct_preds / mnist.test.num_examples))

writer.close()
