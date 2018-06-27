import tensorflow as tf
import numpy as np

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')

x = tf.add(a, b)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
writer.close()

with tf.Session() as sess:
    sess.graph.as_graph_def()

a = [[0, 1], [2, 3], [4, 5]]

with tf.Session() as sess:
    tf.zeros([2, 3]).eval()
    tf.zeros_like(a).eval()
    tf.ones([2, 3]).eval()
    tf.ones_like(a).eval()
    tf.fill([2, 3], 8).eval()
    tf.lin_space(10.0, 13.0, 4).eval()
    tf.range(3, 18, 3).eval()


# initialize

c = tf.Variable(2)

d = tf.get_variable('test', initializer=tf.constant(2))
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))

W = tf.get_variable('big_matrix', shape=(784, 10), initializer=tf.zeros_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer([d, m]))
    sess.run(W.initializer)
    sess.run(W)
    print(W)


my_var = tf.Variable(10)
my_var_op = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var_op)
    sess.run(my_var_op)

my_var = tf.Variable(10)
with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var.assign_add(10))
    sess.run(my_var.assign_sub(2))

W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))
print(sess2.run(W.assign_sub(2)))
sess1.close()
sess2.close()

# g = tf.get_default_graph()
# with g.control_dependencies([a, b, c]):
#     d = ..
#     e = ..

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant(3.0)
c = a + b

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
writer.close()

with tf.Session() as sess:
    sess.run(c, feed_dict={a: [1, 2, 3]})

a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    sess.run(b, feed_dict={a: 15})

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)

x1 = tf.Variable(10, name='x1')
y1 = tf.Variable(20, name='y1')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x1, y1))

