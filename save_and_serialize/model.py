from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ThreeLayerMLP(keras.Model):

  def __init__(self, name=None):
    super(ThreeLayerMLP, self).__init__(name=name)
    self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
    self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
    self.pred_layer = layers.Dense(10, name='predictions')

  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    return self.pred_layer(x)

def get_model():
  return ThreeLayerMLP(name='3_layer_mlp')

model = get_model()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)
# Reset metrics before saving so that loaded model has same state,
# since metric states are not preserved by Model.save_weights
model.reset_metrics()

# Save the model
path = os.path.join(os.getcwd(), 'saved_model')
model.save(path, save_format='tf')
