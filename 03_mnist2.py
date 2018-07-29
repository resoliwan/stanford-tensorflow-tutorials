import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('./data/kaggle_mnist/train.csv', nrows=100)
df.shape
df.describe()

dm = df.as_matrix().astype(np.float32)
dX = dm[:, 1:]
dy = dm[:, 0]
max = (np.max(dy) + 1).astype(np.int32)
donehot_y = np.eye(max)[dy.astype(np.int32)]
dX.shape
donehot_y.shape

batch_size = 10

dataset = tf.data.Dataset.from_tensor_slices((dX, donehot_y))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
X_batch, y_batch = iterator.get_next()

# X_p = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
# y_p = tf.placeholder(tf.int32, [batch_size, 10], name='y_placeholder')


w = tf.get_variable(name='wrights', shape=(784, 10), initializer=tf.random_normal_initializer())
# w = tf.get_variable(name='wrights', shape=(784, 10), initializer=tf.ones_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

y_hat = tf.matmul(X_p, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y_p, name='loss')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

preds = tf.nn.softmax(y_hat)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_p, 1))

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/mnist', sess.graph)
    sess.run(tf.global_variables_initializer())
    for e in range(1):
        t_loss = 0
        sess.run(iterator.initializer)
        try:
            while True:
                _, l = sess.run([optimizer, loss], feed_dict={X_p: X_batch, y_p: y_batch})
                t_loss += l
        except tf.errors.OutOfRangeError:
            pass
        print('t_loss', t_loss)
    writer.close()
