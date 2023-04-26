from benchmarks import get_cifar_based_benchmark

benchmark = get_cifar_based_benchmark(
    scenario_config=f"config_s1.pkl",
    seed=0,
    benchmark=False,
)

_ds = benchmark.train_stream[0].dataset.remove_current_transform_group()
print(_ds[0])
_ds = benchmark.test_stream[0].dataset.remove_current_transform_group()
print(_ds[0])
