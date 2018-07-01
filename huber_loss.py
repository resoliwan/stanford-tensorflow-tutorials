import tensorflow as tf
import pandas as pd
import numpy as np
import math


def huber_loss(y, y_hat, delta):
    diff = math.abs(y - y_hat)
    if diff < delta:
        return 0.5 * diff**2
    else:
        return delta * diff - 0.5 * delta**2


def huber_loass(y, y_hat, delta=1.0):
    residual = tf.abs(y - y_hat)
    def square_f(): return 0.5 * tf.square(residual)
    def abs_f(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, square_f, abs_f)
