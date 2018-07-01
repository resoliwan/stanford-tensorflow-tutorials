import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

df = pd.read_csv('./examples/data/birth_life_2010.txt', delimiter='\t')
df.describe()
dataset = tf.data.Dataset.from_tensor_slices((df['Birthrate'], df['Lifeexpectancy']))
# dataset = dataset.batch(100)
# dataset = dataset.shuffle(buffer_size=100)
iterator = dataset.make_initializable_iterator()
X, y = iterator.get_next()

W = tf.get_variable('W', initializer=tf.constant(1.0, dtype=tf.float64))
b = tf.get_variable('b', initializer=tf.constant(1.0, dtype=tf.float64))
init_op = tf.global_variables_initializer()

y_hat = X * W + b
loss = tf.square((y - y_hat), name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   for i in range(100):
        sess.run(iterator.initializer)
        cost = 0
        try:
            while True:
                # sess.run(iterator.get_next())
                _, out_loss = sess.run([optimizer, loss])
                cost += out_loss
        except tf.errors.OutOfRangeError:
            print(sys.exc_info())
            print('error')
            pass
        print('Epoch {0}: {1}'.format(i, cost))
   w_out, b_out = sess.run([W, b])
   print('w: %f, b: %f' %(w_out, b_out))
        
# plt.scatter(ds[x_label], ds[y_label])
# x_min = ds[x_label].min()
# x_max = ds[x_label].max()
# plt.plot([x_min, x_max], [x_min * w_out + b_out, x_max * w_out + b_out])
# plt.show()
