from __future__ import print_function

import os
import os.path

import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import tensorflow as tf


tf.flags.DEFINE_string("dir_out", "",
                       "Filename of the output .tfrecords file.")
tf.flags.DEFINE_string("npy_path", "", "Name of root file path.")

FLAGS = tf.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

TRAIN_PROP = 0.9
VALIDATION_PROP = 1 - TRAIN_PROP

def main():
    """Main converter function."""
    # Moving MNIST
    from pdb import set_trace; from pprint import pprint; set_trace()
    images = np.load(FLAGS.npy_path)[:,:10,:,:]
    num_examples = images.shape[1]

    full_index = np.arange(num_examples)
    np.random.shuffle(full_index)
    train_index = full_index[:int(TRAIN_PROP * num_examples)]
    validation_index = full_index[int(TRAIN_PROP * num_examples):]

    n_examples_per_file = 10000 // 20
    for set_index, set_name in ((train_index, 'train-mini'),
                                (validation_index, 'validation-mini')):
        filename_prefix = os.path.join(FLAGS.dir_out, set_name)
        for example_idx, i in enumerate(set_index):
            if example_idx % n_examples_per_file == 0:
                file_out = "{}_{:05d}.tfrecords".format(
                    filename_prefix, example_idx // n_examples_per_file)
                print("Writing on:", file_out)
                writer = tf.python_io.TFRecordWriter(file_out)
            if example_idx % 1000 == 0:
                print(example_idx, "/", num_examples)

            sequence_raw = images[:, i, :, :]
            sequence_length, rows, cols = sequence_raw.shape
            depth = 1

            for j in range(sequence_length):
                image_raw = sequence_raw[j]
                random_idx = np.random.randint(0, sequence_length)
                while random_idx == j:
                    random_idx = np.random.randint(0, sequence_length)
                image_2 = sequence_raw[random_idx]
                pair = np.stack([image_raw, image_2])

                from pdb import set_trace; from pprint import pprint; set_trace()

                pair = pair.astype("uint8")
                pair = pair.tostring()

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "height": _int64_feature(rows),
                            "width": _int64_feature(cols),
                            "depth": _int64_feature(depth),
                            "image_pair": _bytes_feature(pair)
                        }
                    )
                )
                writer.write(example.SerializeToString())
            if example_idx % n_examples_per_file == (n_examples_per_file - 1):
                writer.close()
    writer.close()


if __name__ == "__main__":
    main()
