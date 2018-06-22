import numpy as np
import tensorflow as tf

from pprint import pprint

di, dj = 2, 3
dz, dphi = 4, 6
z = np.random.randint(0, 10, size=[di, dz])
phi = np.random.randint(0, 10, size=[dj, dphi])

print('z:')
print(z)
print('phi:')
print(phi)

expected = [[None for _ in range(dj)] for _ in range(di)]
for i in range(di):
  for j in range(dj):
    expected[i][j] = np.concatenate([z[i, :],  phi[j, :]])

expected = np.array(expected)

print('expected:')
pprint(expected)

X, Y = np.meshgrid(range(di), range(dj), indexing='ij')
result_np = np.concatenate([z[X], phi[Y]], axis=2)

print('np result:')
print(result_np)
assert np.all(result_np == expected)

z_tf = tf.constant(z)
phi_tf = tf.constant(phi)

X_tf, Y_tf = tf.meshgrid(
  tf.range(z_tf.shape[0]),
  tf.range(phi_tf.shape[0]),
  indexing='ij')
result_tf = tf.concat(
  [tf.gather(z_tf, X_tf), tf.gather(phi_tf, Y_tf)],
  axis=2)

with tf.Session() as sess:
  result_tf_num = sess.run(result_tf)

print('tf result:')
print(result_tf_num)
assert np.all(result_tf_num == expected)


from pdb import set_trace; from pprint import pprint; set_trace()
