import tensorflow as tf

class Dense(tf.Module):
  def __init__(self, input_size, output_size, name=None):
    super(Dense, self).__init__(name=name)
    self.w = tf.Variable(
        tf.random.normal([input_size, output_size]), name='w')
    self.b = tf.Variable(tf.zeros([output_size]), name='b')

  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


class MLP(tf.Module):
 def __init__(self, input_size, sizes, name=None):
   super(MLP, self).__init__(name=name)
   self.layers = []
   with self.name_scope:
     for size in sizes:
       self.layers.append(Dense(input_size=input_size, output_size=size))
       input_size = size

 @tf.Module.with_name_scope
 def __call__(self, x):
   for layer in self.layers:
     x = layer(x)
   return x

model = MLP(input_size=100, sizes=[30, 30])

tf.saved_model.save(model, "saved")

