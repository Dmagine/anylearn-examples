from anylearn import init_sdk, quick_train

init_sdk('https://anylearn.nelbds.cn', '<username>', '<password>')
train_task, _, _, _ = quick_train(
    algorithm_name="CNN-TensorFlow2-Fashion-MNIST",
    algorithm_dir="./cnn-tensorflow2",
    algorithm_force_update=True,
    dataset_id='DSET2f3215f511ec81fa7225378e3d03', # Fashion MNIST dataset
    dataset_hyperparam_name="data-path",
    entrypoint="python main.py",
    output="./output",
    hyperparams={'batch-size': 128, 'epochs': 3},
    mirror_name="QUICKSTART_TENSORFLOW2.6_CUDA11",
    resource_request=[{
        'default': {
            'CPU': 4,
            'Memory': 8,
        },
    }],
)
print(train_task)
