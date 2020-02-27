import tensorflow
import time

@tensorflow.function
def polar_to_cartesian(r, theta, phi):
    """Convert polar coordinates (r, theta, phi) to cartesian coordinates, (x, y, z)"""
    sin = tensorflow.math.sin
    cos = tensorflow.math.cos
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return (x, y, z)

# Trace polar_to_cartesian.
tensorboard_logdir = "./logs"

#!mkdir -p {tensorboard_logdir}
#!rm -r {tensorboard_logdir}/*

function_name = polar_to_cartesian.__name__

logdir = tensorboard_logdir + "/" + function_name

writer = tensorflow.summary.create_file_writer(logdir)

# Create some phony data to trace.
r, theta, phi = (1.0, 3.14, 1.57)

# Begin trace.
tensorflow.summary.trace_on(graph=True)

# Run the function
x, y, z = polar_to_cartesian(r, theta, phi)

# End trace and write summary.
with writer.as_default():
    tensorflow.summary.trace_export(
      name="polar_to_cartesian",
      step=0,
      profiler_outdir=logdir)
    
time.sleep(2)

#%load_ext tensorboard
#%tensorboard --logdir {tensorboard_logdir}
