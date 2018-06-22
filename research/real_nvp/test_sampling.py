import numpy as np
import tensorflow as tf
from real_nvp_utils import (
    batch_norm, batch_norm_log_diff, conv_layer,
    squeeze_2x2, squeeze_2x2_ordered, standard_normal_ll,
    standard_normal_sample, unsqueeze_2x2, variable_on_cpu)

from pprint import pprint

session = tf.InteractiveSession()

nx, ny = 4,4
nxx, nyy = 2, 2

state_size = nx
context_size = nx**2 - nx

sampled_z = np.arange(2*state_size).reshape(nxx, state_size) * -1
sampled_phi = np.arange(2*nx*ny).reshape(nyy, nx, ny)

X, Y = tf.meshgrid(
  tf.range(sampled_z.shape[0]),
  tf.range(sampled_phi.shape[0]),
  indexing='ij')

sample = tf.matrix_set_diag(
  tf.gather(sampled_phi, Y), tf.gather(sampled_z, X))

sample = tf.reshape(sample, [nxx*nyy] + [nx, ny, 1])

samples_num = session.run(sample)
pprint(samples_num)

assert np.all(samples_num[:, np.arange(nx), np.arange(ny), :] <= 0)
# print(sampled_z)
# print(sampled_phi)

# X, Y = np.meshgrid(
#   np.arange(sampled_z.shape[0]),
#   np.arange(sampled_phi.shape[0]),
#   indexing='ij')

# sample = np.concatenate(
#   [sampled_z[X], sampled_phi[Y]],
#   axis=2).reshape(nxx*nyy, 1, 1, nx*ny)
# pprint(sample)
# from pdb import set_trace; from pprint import pprint; set_trace()
# sample = sample.reshape(4,3,3,1)
# pprint(sample)
# from pdb import set_trace; from pprint import pprint; set_trace()

# # for i in range(2):
# #   assert np.all(sample[i, np.arange(3, np.arange(3), :)] == sampled_z[i])

# # for j in range(2):
# #   assert np.all(sample[j, np.arange(3, np.arange(3), :)] == sampled_z[j])
