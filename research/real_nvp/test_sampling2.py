
import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()
nx, ny = 2, 2
dx, dy = 4, 4

z_out = tf.constant(
  np.arange(nx*ny*dx*dy, dtype=np.float32).reshape(nx*ny, dx, dy, 1))

z_out_pairs_1, z_out_pairs_2 = tf.split(z_out, 2, axis=0)

states_1, contexts_1 = tf.split(z_out_pairs_1, 2, axis=1)
states_2, contexts_2 = tf.split(z_out_pairs_2, 2, axis=1)


state_mean_square_differences = tf.norm(
  (states_1 - states_2), ord='euclidean', axis=(1,2))
context_mean_square_differences = tf.norm(
  (contexts_1 - contexts_2), ord='euclidean', axis=(1,2))

state_mean_square_differences_num = session.run(state_mean_square_differences)
context_mean_square_differences_num = session.run(context_mean_square_differences)


from pdb import set_trace; from pprint import pprint; set_trace()
