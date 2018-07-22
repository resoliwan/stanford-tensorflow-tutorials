import tensorflow as tf

training_dataset = tf.data.Dataset.range(100)
validation_dataset = jk


# https://www.tensorflow.org/guide/dataset://www.tensorflow.org/guide/datasets

iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
X, y = iterator.get_next()

traing_init_op = iterator.make_initializer(training_dataset)
