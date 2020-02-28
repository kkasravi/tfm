import numpy as np
import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100)

    def call(self, inputs):
        outputs = self.dense1(inputs)
        return outputs


model = Model()
optimizer = tf.keras.optimizers.Adam()
writer = tf.summary.create_file_writer("./logdir")


@tf.function
def train(data):
    with tf.name_scope("xxx"):
        with tf.GradientTape() as tape:
            y = model(data)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


x = np.random.rand(10, 100).astype(np.float32)
# y = model(x)
# train.python_function(x)
tf.summary.trace_on()
train(x)
with writer.as_default():
    tf.summary.trace_export("graph", step=0)
    tf.summary.trace_off()
    writer.flush()
