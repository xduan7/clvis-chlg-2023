#######################################
#     DO NOT CHANGE THESE IMPORTS

import sys

sys.path.insert(0, "avalanche")

#######################################

import argparse

import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin, LwFPlugin
from avalanche.training.supervised import Naive
from benchmarks import get_cifar_based_benchmark
from models import SlimResNet18
from strategies.my_plugin import MyPlugin
from strategies.my_strategy import MyStrategy
from utils.competition_plugins import GPUMemoryChecker, RAMChecker, TimeChecker


def main(args):
    # --- Device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- Benchmark
    benchmark = get_cifar_based_benchmark(
        scenario_config=args.config_file,
        seed=args.seed,
        benchmark=args.benchmark,
    )

    # --- Model
    model = SlimResNet18(n_classes=benchmark.n_classes)

    # --- Logger and metrics
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # --- Competition Plugins
    # DO NOT REMOVE OR CHANGE THESE PLUGINS:
    competition_plugins = [
        # GPUMemoryChecker(max_allowed=1000, device=device),
        # RAMChecker(max_allowed=3200),
        TimeChecker(max_allowed=500),
    ]

    # --- Your Plugins
    plugins = [EWCPlugin(ewc_lambda=0.5), LwFPlugin(alpha=1.0), MyPlugin()]

    # --- Strategy
    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=64,
        train_epochs=args.num_epochs,
        eval_mb_size=100,
        device=device,
        plugins=competition_plugins + plugins,
        evaluator=eval_plugin,
    )

    # --- Training Loops
    for experience in benchmark.train_stream:
        cl_strategy.train(experience, num_workers=args.num_workers)

    # --- Make prediction on test-set samples
    predictions = predict_test_set(
        cl_strategy.model, benchmark.test_stream[0].dataset, device
    )

    # Save predictions
    if args.benchmark:
        output_name = f"{args.name}_{args.config}.npy"
        np.save(output_name, predictions)


def predict_test_set(model, test_set, device):
    print("Making prediction on test-set samples")

    model.eval()
    dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
    preds, trgts = [], []
    with torch.no_grad():
        for x, y, _ in dataloader:
            pred = model(x.to(device)).detach().cpu()
            preds.append(pred)
            trgts.append(y)

    trgts = torch.cat(trgts, dim=0).numpy()
    preds = torch.cat(preds, dim=0)
    preds = torch.argmax(preds, dim=1).numpy()
    print(f"Test-set accuracy: {np.mean(preds == trgts)}")

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Name of the experiment. Used to create the output files.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--config",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Select the scenario configuration. ",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Use benchmark (challenge) datasets, in which case, the test "
        "results will be unavailable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs for the representation learning phase.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    args.config_file = f"config_s{args.config}.pkl"
    main(args)
