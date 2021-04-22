import os
import gzip

import numpy as np
import tensorflow as tf


def load_data_local(path='data/fashion',
                    files=[
                        'train-labels-idx1-ubyte.gz',
                        'train-images-idx3-ubyte.gz',
                        't10k-labels-idx1-ubyte.gz',
                        't10k-images-idx3-ubyte.gz'
                    ]):
    paths = [os.path.join(path, file) for file in files]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def build_tf_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(X[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(y, tf.int64)
        )
    )
    return dataset.shuffle(len(y)).batch(batch_size)
