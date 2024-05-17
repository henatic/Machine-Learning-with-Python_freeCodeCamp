from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from six.moves import urllib

import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)
eval_dataset = tf.data.Dataset.from_tensor_slices(
      (eval_features, eval_labels)).batch(1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)

model.compile(optimizer=optimizer, loss="mse")