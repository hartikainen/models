import tensorflow as tf
import numpy as np
from pprint import pprint

from real_nvp_utils import (
    batch_norm, batch_norm_log_diff, conv_layer,
    squeeze_2x2, squeeze_2x2_ordered, standard_normal_ll,
    standard_normal_sample, unsqueeze_2x2, variable_on_cpu)

n_scale = 4


z_complete = tf.constant((np.arange(64) + 1).reshape(1, 8, 8, 1), dtype=tf.float32)
z_compressed_list = [z_complete]
z_noisy_list = [z_complete]
z_lost = z_complete

for scale_idx in range(n_scale - 1):
    z_lost = squeeze_2x2_ordered(z_lost)
    z_lost, _ = tf.split(axis=3, num_or_size_splits=2, value=z_lost)
    z_compressed = z_lost
    z_noisy = z_lost
    for _ in range(scale_idx + 1):
        z_compressed = tf.concat(
          [z_compressed, tf.zeros_like(z_compressed)], 3)
        z_compressed = squeeze_2x2_ordered(
          z_compressed, reverse=True)
        z_noisy = tf.concat(
          [z_noisy, tf.random_normal(
            z_noisy.get_shape().as_list())], 3)
        z_noisy = squeeze_2x2_ordered(z_noisy, reverse=True)
    z_compressed_list.append(z_compressed)
    z_noisy_list.append(z_noisy)


with tf.Session() as sess:
  compresseds, noisys = sess.run([z_compressed_list, z_noisy_list])

from pdb import set_trace; from pprint import pprint; set_trace()

z_out = tf.concat([
  (np.arange(18, dtype=np.float32) + 1).reshape(2, 3, 3, 1),
  (np.arange(18, dtype=np.float32)).reshape(2, 3, 3, 1)
], axis=0)

final_shape = [3,3,1]

z_out_flat = tf.reshape(z_out, (-1, tf.reduce_prod(final_shape)))
z_out_pairs_1, z_out_pairs_2 = tf.split(z_out, 2, axis=0)

state_size = 3
context_size = np.prod(final_shape) - state_size
# state = z_out[:,:,:, :state_size]
states_1 = tf.matrix_diag_part(z_out_pairs_1[:,:,:,0])[:, :state_size]
states_2 = tf.matrix_diag_part(z_out_pairs_2[:,:,:,0])[:, :state_size]

state_sum_square_differences = tf.reduce_sum(
  (states_1 - states_2) ** 2, axis=1)

from pdb import set_trace; from pprint import pprint; set_trace()
context_sum_square_differences = tf.reduce_sum(
  (z_out_pairs_1 - z_out_pairs_2) ** 2,
  axis=list(range(1, z_out_pairs_1.shape.ndims))
) - state_sum_square_differences

state_mean_square_differences = (
  state_sum_square_differences / state_size)
context_mean_square_differences = (
  context_sum_square_differences / context_size)

# states_1 = tf.matrix_diag_part(z_out_pairs_1[:,:,:,0])[:, :state_size]
# states_2 = tf.matrix_diag_part(z_out_pairs_2[:,:,:,0])[:, :state_size]

# state_square_differences = tf.reduce_mean(
#   (states_1 - states_2) ** 2, axis=1)

# context_1 = tf.matrix_set_diag(
#   z_out_pairs_1[:,:,:,0],
#   tf.zeros((z_out_pairs_1.shape[0], z_out_pairs_1.shape[1]), dtype=tf.float32))
# context_2 = tf.matrix_set_diag(
#   z_out_pairs_2[:,:,:,0],
#   tf.zeros((z_out_pairs_2.shape[0], z_out_pairs_2.shape[1]), dtype=tf.float32))

# context_square_differences = tf.reduce_mean(
#   (context_1 - context_2) ** 2, axis=list(range(1, context_1.shape.ndims)))

from pdb import set_trace; from pprint import pprint; set_trace()

with tf.Session() as sess:
  print('z_out')
  pprint(sess.run(z_out))
  print("[state_sum_square_differences, context_sum_square_differences]")
  pprint(sess.run([state_sum_square_differences, context_sum_square_differences]))
  print("[state_mean_square_differences, context_mean_square_differences]")
  pprint(sess.run([state_mean_square_differences, context_mean_square_differences]))
