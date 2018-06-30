import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('./examples/data/birth_life_2010.txt', delimiter='\t')
df.describe()
df.shape[0]
input_label = 'Birthrate'
target_label = 'Lifeexpectancy'

X = tf.placeholder(tf.float32, name='X')
y = tf.placeholder(tf.float32, name='y')
W = tf.get_variable('w', initializer=tf.constant(0.0))
b = tf.get_variable('b', initializer=tf.constant(0.0))
init_variables = tf.global_variables_initializer()

y_hat = tf.multiply(W, X) + b
loss = tf.sqrt((y - y_hat)**2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

with tf.Session() as sess:
    sess.run(init_variables)
    writer = tf.summary.FileWriter('./graphs/linear_reg2', sess.graph)
    for batch in range(400):
        cost = 0
        for i in range(df.shape[0]):
            _, loss_out = sess.run([optimizer, loss], feed_dict={X: df[input_label].loc[i], y: df[target_label][i]})
            cost += loss_out
        W_out, b_out = sess.run([W, b])
        print(W_out, b_out, cost)
    writer.close()


plt.scatter(df[input_label], df[target_label])
x_min = df[input_label].min()
x_max = df[input_label].max()
plt.plot([x_min, x_max], [W_out * x_min + b_out, W_out * x_max + b_out])
plt.show()





