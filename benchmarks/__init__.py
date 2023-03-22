import dill

import avalanche.benchmarks
from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.datasets import CIFAR100


from .cir_benchmark import generate_benchmark


def get_cifar_based_benchmark(scenario_config, seed):
    # Download dataset if not available
    _ = CIFAR100(root="./data/datasets", train=True,
                 transform=None, download=True)
    _ = CIFAR100(root="./data/datasets", train=False,
                 transform=None, download=True)

    # Load challenge datasets
    with open("./data/challenge_train_set.pkl", "rb") as pkl_file:
        train_set = dill.load(pkl_file)

    with open("./data/challenge_test_set.pkl", "rb") as pkl_file:
        test_set = dill.load(pkl_file)

    # avalanche.benchmarks.utils.classification_dataset.ClassificationDataset
    # Redefine train and test sets, using basic CIFAR100 datasets
    _exp = SplitCIFAR100(
        n_experiences=1,
        return_task_id=False,
        shuffle=True,
        dataset_root="./data/datasets",
    )
    train_set = _exp.train_stream[0].dataset
    test_set = _exp.test_stream[0].dataset

    # Load scenario config
    with open(f"./scenario_configs/{scenario_config}", "rb") as pkl_file:
        scenario_config = dill.load(pkl_file)

    # Benchmarks
    benchmark = generate_benchmark(seed=seed, train_set=train_set,
                                   test_set=test_set, **scenario_config)

    return benchmark


__al__ = ["get_cifar_based_benchmark"]