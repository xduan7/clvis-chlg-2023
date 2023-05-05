#######################################
#     DO NOT CHANGE THESE IMPORTS

import sys

sys.path.insert(0, "avalanche")

#######################################

import argparse

import numpy as np
import torch.optim.lr_scheduler
from sklearn.metrics import top_k_accuracy_score

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from benchmarks import get_cifar_based_benchmark
from hat import HATConfig
from models import *
from strategies.classification import Classification
from strategies.sup_contrast import SupContrast
from utils.competition_plugins import GPUMemoryChecker, RAMChecker, TimeChecker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
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
        required=True,
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
        "--init_method",
        type=str,
        default="none",
        choices=["none", "kaiming_normal", "orthogonal"],
        help="Initialization method for the model.",
    )
    parser.add_argument(
        "--zero_init_last",
        action="store_true",
        help="Initialize the last layer batchnorm weight with zeros in a "
        "residual block.",
    )
    parser.add_argument(
        "--hat",
        action="store_true",
        help="Use hard attention to the task.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision.",
    )
    parser.add_argument(
        "--rep_num_epochs",
        type=int,
        default=20,
        help="Number of epochs for the representation learning phase.",
    )
    parser.add_argument(
        "--rep_lr",
        type=float,
        default=0.01,
        help="Learning rate for the representation learning phase.",
    )
    parser.add_argument(
        "--rep_batch_size",
        type=int,
        default=32,
        help="Batch size for the representation learning phase.",
    )
    parser.add_argument(
        "--rep_proj_head_dim",
        type=int,
        default=128,
        help="Dimension of the projection head for the representation "
        "learning phase.",
    )
    parser.add_argument(
        "--clf_num_epochs",
        type=int,
        default=20,
        help="Number of epochs for the classification learning phase.",
    )
    parser.add_argument(
        "--clf_lr",
        type=float,
        default=0.001,
        help="Learning rate for the classification learning phase.",
    )
    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=32,
        help="Batch size for the classification learning phase.",
    )
    parser.add_argument(
        "--clf_freeze_backbone",
        action="store_true",
        help="Freeze the backbone during the classification learning phase.",
    )
    parser.add_argument(
        "--clf_train_exp_logits_only",
        action="store_true",
        help="Train only the classification logits that are present in "
        "the current experience, so that the other logits are frozen.",
    )
    parser.add_argument(
        "--clf_logit_calibr",
        type=str,
        default="norm",
        choices=["none", "temp", "norm"],
        help="Logit calibration method.",
    )
    parser.add_argument(
        "--clf_replay",
        action="store_true",
        help="Use a replay strategy for the classification learning phase.",
    )
    parser.add_argument(
        "--clf_lr_decay",
        action="store_true",
        help="Use a learning rate decay for the classification learning "
        "phase. Not configurable at the moment.",
    )
    parser.add_argument(
        "--tst_time_aug",
        type=int,
        default=1,
        help="Number of test-time augmentations. 1 means no augmentation.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    args.config_file = f"config_s{args.config}.pkl"

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
    num_classes = benchmark.n_classes
    # Need an extra class for "out-of-experience" samples if we use replay
    # and train only the logits of the current experience
    if args.clf_replay and args.clf_train_exp_logits_only:
        num_classes += 1
    if args.hat:
        hat_config = HATConfig(
            num_tasks=benchmark.n_experiences,
            max_trn_mask_scale=100,
            init_strat="dense",

        )
        model = HATSlimResNet18(
            n_classes=num_classes,
            hat_config=hat_config,
        )
    else:
        hat_config = None
        model = SlimResNet18(n_classes=num_classes)
    init_resnet(
        model,
        method=args.init_method,
        zero_init_last=args.zero_init_last,
    )
    model.to(device)

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
        TimeChecker(max_allowed=500) if args.benchmark else TimeChecker(),
    ]

    # --- Your Plugins
    # plugins = [EWCPlugin(ewc_lambda=0.5), LwFPlugin(alpha=1.0), MyPlugin()]

    # --- Strategy
    rep = SupContrast(
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.rep_lr),
        train_mb_size=args.rep_batch_size,
        train_epochs=args.rep_num_epochs,
        device=device,
        proj_head_dim=args.rep_proj_head_dim,
        plugins=[*competition_plugins],
        verbose=args.verbose,
    )
    clf = Classification(
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.clf_lr),
        freeze_backbone=args.clf_freeze_backbone,
        train_exp_logits_only=args.clf_train_exp_logits_only,
        train_mb_size=args.clf_batch_size,
        device=device,
        train_epochs=args.clf_num_epochs,
        logit_calibr=args.clf_logit_calibr,
        replay=args.clf_replay,
        lr_decay=args.clf_lr_decay,
        plugins=[
            *competition_plugins,
        ],
        verbose=args.verbose,
    )

    # --- Training Loops
    for __exp in benchmark.train_stream:
        print(f"Training on experience {__exp.current_experience} ... ")
        rep.train(
            experiences=__exp,
            num_workers=args.num_workers,
        )
        clf.train(
            experiences=__exp,
            num_workers=args.num_workers,
        )
        # TODO: clustering for prediction
    print(f"Training done in {competition_plugins[0].time_spent:.2f} minutes.")

    # --- Make prediction on test-set samples
    tst_targets, tst_predictions, tst_logits, tst_features = clf.predict(
        benchmark.test_stream[0].dataset,
        tst_time_aug=args.tst_time_aug,
    )

    # Save predictions or print the results
    if args.benchmark:
        import dill

        with open("./data/challenge_test_labels.pkl", "rb") as __f:
            tst_labels = dill.load(__f)

        tst_labels = np.array(tst_labels)
        print(f"Test-set accuracy: {np.mean(tst_predictions == tst_labels)}")

        np.save(f"{args.name}_{args.config}_logits.npy", tst_logits)
        np.save(f"{args.name}_{args.config}_predictions.npy", tst_predictions)
    else:
        print(f"Test-set accuracy: {np.mean(tst_predictions == tst_targets)}")

    # Evaluate top-5 accuracy based on logits and targets
    top_5_acc = top_k_accuracy_score(
        y_score=tst_logits.mean(0),
        y_true=tst_targets,
        k=5,
    )
    print(f"Test-set top-5 accuracy: {top_5_acc}")

    # Plot the confusion matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    plt.figure(figsize=(10, 10))
    plt.imshow(
        confusion_matrix(
            y_true=tst_targets,
            y_pred=tst_predictions,
        ),
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Count the predictions occurrences
    import collections

    pred_counter = collections.Counter(tst_predictions)
    print(pred_counter)

    # Find patterns in mis-classifications
    cm = confusion_matrix(
        y_true=tst_targets,
        y_pred=tst_predictions,
    )
    cls_to_last_seen_exp = {}
    for __exp, __cls_in_exp in enumerate(clf.classes_in_experiences):
        for __cls in __cls_in_exp:
            cls_to_last_seen_exp[__cls] = __exp
    # Get the highest N values (could be on the diagonal)
    n = 50
    # Get the indices of the N highest values in the confusion matrix
    indices = np.argpartition(cm.ravel(), -n)[-n:]
    # Sort the indices
    indices = indices[np.argsort(-cm.ravel()[indices])]
    for __i in indices:
        __r, __c = np.unravel_index(__i, cm.shape)
        if __r != __c:
            __re = cls_to_last_seen_exp[__r]
            __ce = cls_to_last_seen_exp[__c]

            __shared_exp_id = None

            for __id, __exp in enumerate(benchmark.train_stream):
                if (
                    __r in __exp.classes_in_this_experience
                    and __c in __exp.classes_in_this_experience
                ):
                    if __shared_exp_id is None:
                        __shared_exp_id = [__id]
                    else:
                        __shared_exp_id.append(__id)

            if __shared_exp_id is not None:
                print(
                    f"Class {__r} (last seen in {__re}) is "
                    f"misclassified as {__c} (last seen in {__ce}) "
                    f"{cm[__r, __c]} times "
                    f"and they are in the same exp {__shared_exp_id}"
                )
            else:
                print(
                    f"Class {__r} (last seen in {__re}) is "
                    f"misclassified as {__c} (last seen in {__ce}) "
                    f"{cm[__r, __c]} times "
                    f"and they are never in the same exp"
                )

# Note to self
# These three experiments are all performed with the new cosine scaling
# New scaling has a hard lower bound of 1.0

# `train` is 20/20: 0.3003/0.3556
# `train(1)` is 20/20 with lr decay during clf 0.2807/0.3283
# `train(2)` is 10/20 with lr decay during clf: 0.2453

# `train(3)` is 20/20 with rep batch size 64 0.2986/0.3418
# `train(4)` is 20/20 with clf batch size 64 0.2889/0.3301
# `train(5)` is 20/20 with rep batch size 64 and clf batch size 64 0.2939/0.337

# the first instance on Kontrol is the replay with 2 samples that are
# closest to the mean 0.0494
# the second instance on Kontrol is the replay with 2 samples selected with
# K-means 0.0629

# `train(6)` is 20/20 is the replay with 2 samples that are closest to the
#   mean with non augmented dataloader during replay sampling 0.0494
# `train(7)` is 20/20 is the replay with 2 samples selected with K-means with
#   non augmented dataloader during replay sampling 0.0629 <- this is the way

# `train` is default to test out if anything is broken 0.3003/0.3558
# `train(1)` is default with replay 0.299/0.3377

# `train(2)` is default with amp 0.293/0.3398

# `train(3)` is default on config 3 0.2388/0.2839
# `train(4)` is default on config 3 with replay 0.2426/0.2952

# TODO: nvidia apex
# TODO: fuzzy replay
# TODO: replay only when the number of samples is low

# `train(5)` is default on config 1 with replay and resetting the last clf
#   neuron between experiences 0.2992/0.3521
# `train(6)` is default on config 3 with replay and resetting the last clf
#   neuron between experiences and lower bound for replay 0.2442/0.2884
# `train(7)` is default on config 3 with replay and resetting the last clf
#   neuron between experiences and lower bound for replay and rand noise for
#   replay logits 0.2396/0.2848

# TODO: label smoothing
# TODO: different optimizers (and arguments)
