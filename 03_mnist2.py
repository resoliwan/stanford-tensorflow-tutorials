import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('./data/kaggle_mnist/train.csv', nrows=10)
df.describe()

ds = df.as_matrix()

train_m = df.shape[0]

ds.shape

data = np.arange(10).reshape(2, 5)
data[:, -1]
data[0, 0:-1]

max = np.max(data[:, -1]) + 1
np.eye(max)[data[:, -1]]


x = ds[:, 0:-1]
assert x.shape == (train_m, 783)

y = ds[:, -1]


