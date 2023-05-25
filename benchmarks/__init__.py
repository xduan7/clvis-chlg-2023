import os.path

import dill
import torch

from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.datasets import CIFAR100

from .cir_benchmark import generate_benchmark


def get_cifar_based_benchmark(scenario_config_file, seed):
    # Load scenario config
    with open(scenario_config_file, "rb") as pkl_file:
        scenario_config = dill.load(pkl_file)

    # Download dataset if not available
    _ = CIFAR100(
        root="./data/datasets", train=True, transform=None, download=True
    )
    _ = CIFAR100(
        root="./data/datasets", train=False, transform=None, download=True
    )

    # Load challenge datasets
    with open("./data/challenge_train_set.pkl", "rb") as pkl_file:
        train_set = dill.load(pkl_file)

    with open("./data/challenge_test_set.pkl", "rb") as pkl_file:
        test_set = dill.load(pkl_file)

    # Benchmarks
    benchmark = generate_benchmark(
        seed=seed,
        train_set=train_set,
        test_set=test_set,
        **scenario_config,
    )

    return benchmark


__al__ = ["get_cifar_based_benchmark"]
