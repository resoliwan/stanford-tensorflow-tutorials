import tensorflow as tf
import pandas as pd
import numpy as np
from examples import utils
import matplotlib.pyplot as plt

ds = pd.read_csv('./examples/data/birth_life_2010.txt', delimiter='\t')
ds.head()
ds.describe()
x_label = 'Birthrate'
y_label = 'Lifeexpectancy'
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

w = tf.get_variable('weights', initializer=tf.constant(1.0))
b = tf.get_variable('bias', initializer=tf.constant(1.0))

y_hat = w * x + b
loss = tf.square(y - y_hat, name='loss')
init_op = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

ds['Birthrate'][1]
ds['Lifeexpectancy'][1]


w_out = 0
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
    for i in range(100):
        cost = 0
        for i in range(190):
            _, l = sess.run([optimizer, loss], feed_dict={x: ds[x_label][i], y: ds[y_label][i]})
            # print('l', l)
            cost += l
            # print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
        print('cost', cost)
    w_out, b_out = sess.run([w, b])
    print('w_out', w_out)
    print('b_out', b_out)
    writer.close()

plt.scatter(ds[x_label], ds[y_label])
x_min = ds[x_label].min()
x_max = ds[x_label].max()
plt.plot([x_min, x_max], [x_min * w_out + b_out, x_max * w_out + b_out])
plt.show()
