import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('./examples/data/birth_life_2010.txt', delimiter='\t')
df.describe()
dataset = tf.data.Dataset.from_tensor_slices((df['Birthrate'], df['Lifeexpectancy']))
iterator = dataset.make_initializable_iterator()
X, y = iterator.get_next()

# with tf.Session() as sess:
#     sess.run(iterator.initializer)
#     sess.run(iterator.get_next())

W = tf.get_variable('weights', initializer=tf.constant(0.0, dtype=tf.float64))
b = tf.get_variable('bias', initializer=tf.constant(0.0, dtype=tf.float64))
y_hat = W * X + b
loss = tf.square(y - y_hat)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        sess.run(iterator.initializer)
        cost = 0
        try:
            while True:
                _, out_loss = sess.run([optimizer, loss])
                cost += out_loss
        except:
            pass
        out_W, out_b = sess.run([W, b])
        print('epoch %d out_W %f out_b %f cost %f' % (epoch, out_W, out_b, cost))


x_label='Birthrate'
y_label='Lifeexpectancy'
plt.scatter(df[x_label], df[y_label])
x_min = df[x_label].min()
x_max = df[x_label].max()
plt.plot([x_min, x_max], [x_min * out_W + out_b, x_max * out_W + out_b])
plt.show()



