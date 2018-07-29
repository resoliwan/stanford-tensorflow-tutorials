import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

batch_size = 10
verification_size = 100
train_size = 50000
n_epochs = 50

test_df = pd.read_csv('./data/kaggle_mnist/test.csv')
test_dX = test_df.as_matrix().astype(np.float32)
test_da = tf.data.Dataset.from_tensor_slices((test_dX))
test_da = test_da.batch(batch_size)

test_iterator = test_da.make_one_shot_iterator()
test_X = test_iterator.get_next()

def get_datset(filename, batch_size, skiprows, nrows):
    df = pd.read_csv(filename, skiprows=skiprows, nrows=nrows)
    dm = df.as_matrix().astype(np.float32)
    dX = dm[:, 1:]
    dy = dm[:, 0]
    max = (np.max(dy) + 1).astype(np.int32)
    donehot_y = np.eye(max)[dy.astype(np.int32)]
    ds = tf.data.Dataset.from_tensor_slices((dX, donehot_y))
    return ds.batch(batch_size)

verification_ds = get_datset('./data/kaggle_mnist/train.csv', batch_size, 0, verification_size)
train_ds = get_datset('./data/kaggle_mnist/train.csv', batch_size, verification_size, train_size)

iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
train_init = iterator.make_initializer(train_ds)
verification_init = iterator.make_initializer(verification_ds)
X_batch, y_batch = iterator.get_next()

W = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

y_hat = tf.matmul(X_batch, W) + b

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y_batch, name='loss')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

preds = tf.nn.softmax(y_hat)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_batch, 1))
correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

test_y_hat = tf.matmul(test_X, W) + b
predicts = tf.argmax(tf.nn.softmax(test_y_hat), 1)

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs/minit3', sess.graph)
    sess.run(tf.global_variables_initializer())
    for e in range(n_epochs):
        # sess.run(iterator.initializer)
        sess.run(train_init)
        try:
            while True:
                _, out_l = sess.run([optimizer, loss])
                print('out_l', out_l)
        except tf.errors.OutOfRangeError:
            pass
    sess.run(verification_init)
    total_correct_preds = 0;
    try:
        while True:
            out_correct_preds, out_correct = sess.run([correct_preds, correct])
            total_correct_preds += out_correct
    except tf.errors.OutOfRangeError:
            pass
    # writer.close()
    print('Accuracy {0}'.format(total_correct_preds/verification_size))
    result = []
    try: 
        while True:
            out_predicts = sess.run([predicts])
            result.append(out_predicts)
            # print('out_predicts', out_predicts)
    except tf.errors.OutOfRangeError:
            pass

labels = np.asarray(result).flatten()

sample_df = pd.read_csv('./data/kaggle_mnist/sample_submission.csv')
sub_df = pd.DataFrame({'ImageId': sample_df.ImageId, 'Label': labels})
sub_df.to_csv('./data/kaggle_mnist/submission1.csv', index=False)
