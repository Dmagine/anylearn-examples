import argparse
import tensorflow as tf
from numpy.random import seed

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

tf.keras.backend.manual_variable_initialization(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--data-path', type=str, default="./data/fashion")
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    assert args.model_dir

    # Load raw data
    _, (X_test, y_test) = load_data_local(path=args.data_path)
    dataset = build_tf_dataset(X_test, y_test, args.batch_size)

    # Load model
    model = tf.keras.models.load_model(args.model_dir)

    # Eval
    result = model.evaluate(dataset, verbose=2)
    with open("result.txt", "w") as f:
        f.write(str(result))
