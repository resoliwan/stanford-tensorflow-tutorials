import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv('./data/test.csv')
df.describe()

dm = df.as_matrix()
y = dm[:, 0]
x = dm[:, 1:]

max = np.max(dm[:, 0]) + 1

onehot_y = np.eye(max)[dm[:, 0]]

dataset = tf.data.Dataset.from_tensor_slices((x, onehot_y))
iterator = dataset.make_initializable_iterator()
X, y = iterator.get_next()

with tf.Session() as sess:
    for e in range(3):
        sess.run(iterator.initializer)
        try:
            while True:
                X_out, y_out = sess.run([X, y])
                print('X_out', X_out)
                print('y_out', y_out)
        except tf.errors.OutOfRangeError:
            print('OutOfRangeError')
            pass
