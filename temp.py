import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('./examples/data/birth_life_2010.txt', delimiter='\t')
dataset.describe()
x_label = 'Birthrate'
y_label = 'Lifeexpectancy'

dataset[0:1]
dataset[x_label][0]
dataset[y_label][0]

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

w = tf.get_variable('weights', initializer=tf.constant(1.0))
b = tf.get_variable('bias', initializer=tf.constant(1.0))
init_op = tf.global_variables_initializer()

y_hat = w * x + b
loss = tf.square(y - y_hat, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./graphs/linear_reg6', sess.graph)
    for l in range(100):
        cost = 0
        for i in range(190):
            _, l = sess.run([optimizer, loss], feed_dict={x: dataset[x_label][i], y: dataset[y_label][i]})
            cost += l
        print('cost', cost)
    w_out, b_out = sess.run([w, b])
    print('w_out', w_out)
    print('b_out', b_out)
    writer.close()






