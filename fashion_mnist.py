import argparse
import time

import tensorflow as tf
from numpy.random import seed
import anylearn

from dataset import load_data_local, build_tf_dataset

# GPU restriction
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# TF global random seed
tf.random.set_seed(860597652)
# Numpy global random seed
seed(860597652)


class EpochEndCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs['val_acc'] if 'val_acc' in logs else logs['val_accuracy']
        print(f"Current acc: {acc}")
        anylearn.report_intermediate_metric(acc)


if __name__ == '__main__':
    t = int(time.time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="./data/fashion")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    _batch_size = args.batch_size
    _learning_rate = args.lr
    _epochs = args.epochs
    _log_dir = f'./logs'
    _ckpt_dir = f'./ckpt'
    _model_dir = f'./model'

    # Load raw data
    (X_train, y_train), (X_test, y_test) = load_data_local(path=args.data_path)
    dset_train = build_tf_dataset(X_train, y_train, _batch_size)
    dset_test = build_tf_dataset(X_test, y_test, _batch_size)

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(_learning_rate)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(_ckpt_dir),
        tf.keras.callbacks.TensorBoard(_log_dir),
        EpochEndCallback(),
    ]

    # Train
    op = time.time()
    model.fit(dset_train,
              callbacks=callbacks,
              epochs=_epochs,
              validation_data=dset_test,
              verbose=1)
    ed = time.time()

    # Test
    test_loss, test_acc = model.evaluate(dset_test, verbose=2)
    anylearn.report_final_metric(test_acc)

    # Save
    model.save(_model_dir)
