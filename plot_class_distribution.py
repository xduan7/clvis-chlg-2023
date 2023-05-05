"""Plot the class distribution of three configurations"""


import matplotlib.pyplot as plt

from benchmarks import get_cifar_based_benchmark

# 9 subplots in horizontal direction
fig, axs = plt.subplots(9, 1, figsize=(16, 16))

for __c in [1, 2, 3]:
    _benchmark = get_cifar_based_benchmark(
        scenario_config=f"config_s{__c}.pkl",
        seed=0,
        benchmark=True,
    )
    _dataset = _benchmark.train_stream[0].dataset
    _targets = [__s[1] for __s in _dataset]

    from collections import Counter

    # For each benchmark, make three subplots
    # The x-axis is shared, representing experiences in the benchmark
    # The first one is the number of samples
    # The second one is the number of classes in the experience
    # The third one is the number of samples per class in the experience,
    # where each class is represented by a different color, if present
    # in the experience
    _num_samples = []
    _num_classes = []
    _classes = []
    for __experience in _benchmark.train_stream:
        _num_samples.append(len(__experience.dataset))
        _num_classes.append(
            len(__experience.classes_trained_in_this_experience)
        )
        _classes.append(__experience.classes_trained_in_this_experience)

    # Plot the number of samples
    axs[(__c - 1) * 3].plot(_num_samples)
    axs[(__c - 1) * 3].set_title(f"Config {__c}")
    axs[(__c - 1) * 3].set_xlabel("Experience")
    axs[(__c - 1) * 3].set_ylabel("Number of samples")

    # Plot the number of classes
    axs[(__c - 1) * 3 + 1].plot(_num_classes)
    axs[(__c - 1) * 3 + 1].set_title(f"Config {__c}")
    axs[(__c - 1) * 3 + 1].set_xlabel("Experience")
    axs[(__c - 1) * 3 + 1].set_ylabel("Number of classes")

    # Plot the class presence over the experiences with scatter plot
    _class_presence = []
    for __i, __classes in enumerate(_classes):
        for __j in __classes:
            _class_presence.append([__i, __j])
    axs[(__c - 1) * 3 + 2].scatter(
        [__p[0] for __p in _class_presence],
        [__p[1] for __p in _class_presence],
    )
    axs[(__c - 1) * 3 + 2].set_title(f"Config {__c}")
    axs[(__c - 1) * 3 + 2].set_xlabel("Experience")
    axs[(__c - 1) * 3 + 2].set_ylabel("Class presence")

plt.tight_layout()
plt.show()
