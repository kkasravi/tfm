import tensorflow as tf
import tensorflow_datasets as tfds
import datetime as dt
import pdb

mean_metric = tf.keras.metrics.Mean()
mean_metric.update_state(2.0)
mean_metric.update_state(3.0)
mean_metric.update_state(4.0)
print(mean_metric.result().numpy())


mean_metric.reset_states()
print(mean_metric.result().numpy())

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
BATCH_SIZE=64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.map(lambda x, y: (tf.expand_dims(x, -1) / 255.0, y))
train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.map(lambda x, y: (tf.expand_dims(x, -1) / 255.0, y))
valid_dataset = valid_dataset.repeat()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, 2, 1, activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, 2, 1, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))

class MetricLayer(tf.keras.layers.Layer):
  def __init__(self, layer_to_log):
    super(MetricLayer, self).__init__()
    self.layer_to_log = layer_to_log
    
  def call(self, input):
    self.add_metric(tf.keras.backend.std(self.layer_to_log.variables[0]),
                    name=f'std_of_{self.layer_to_log.name}_kernel',
                    aggregation='mean')
    return input


metric_model = tf.keras.Sequential()
metric_model.add(model)
metric_model.add(MetricLayer(model.layers[0]))

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), update_freq='batch')
]

metric_model.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


metric_model.fit(train_dataset, steps_per_epoch=10000//BATCH_SIZE, epochs=5,
                 validation_data=valid_dataset, validation_steps=5,
                 callbacks=callbacks)
