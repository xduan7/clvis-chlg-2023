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
from utils.params import merge_params, print_params
from utils.lars import LARS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="hat_with_rep",
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
        "--hat_grad_comp_factor",
        type=float,
        default=100.0,
        help="Gradient compensation factor for the HAT masks.",
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
        "--rep_hat_reg_base_factor",
        type=float,
        default=1.0,
        help="Base factor for the HAT regularization term for the "
        "representation learning phase.",
    )
    parser.add_argument(
        "--rep_num_replay_samples_per_batch",
        type=int,
        default=0,
        help="Number of replay samples per batch for the representation "
        "learning phase.",
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
        "--clf_hat_reg_base_factor",
        type=float,
        default=1.0,
        help="Base factor for the HAT regularization term for the "
        "classification learning phase.",
    )
    parser.add_argument(
        "--clf_num_replay_samples_per_batch",
        type=int,
        default=0,
        help="Number of replay samples per batch for the classification "
        "learning phase.",
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
    try:
        import nni

        args = merge_params(args, nni.get_next_parameter())
    except ImportError:
        pass
    print_params(args)

    # --- Device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    if device.type == "cuda":
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True

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
    if (
        args.clf_num_replay_samples_per_batch > 0
        and args.clf_train_exp_logits_only
    ):
        num_classes += 1
    if args.hat:
        hat_config = HATConfig(
            num_tasks=benchmark.n_experiences,
            max_trn_mask_scale=100,
            init_strat="dense",
            grad_comp_factor=args.hat_grad_comp_factor,
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
        # GPUMemoryChecker(max_allowed=4000, device=device),
        # RAMChecker(max_allowed=4000),
        TimeChecker(max_allowed=500)
        if args.benchmark
        else TimeChecker(),
    ]

    # --- Your Plugins
    # plugins = [EWCPlugin(ewc_lambda=0.5), LwFPlugin(alpha=1.0), MyPlugin()]

    # --- Strategy
    rep = SupContrast(
        model,
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=args.rep_lr,
            amsgrad=True,
        ),
        train_mb_size=args.rep_batch_size,
        train_epochs=args.rep_num_epochs,
        hat_reg_base_factor=args.rep_hat_reg_base_factor,
        num_replay_samples_per_batch=args.rep_num_replay_samples_per_batch,
        device=device,
        proj_head_dim=args.rep_proj_head_dim,
        plugins=[*competition_plugins],
        verbose=args.verbose,
    )
    clf = Classification(
        model,
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=args.clf_lr,
            amsgrad=True,
        ),
        freeze_backbone=args.clf_freeze_backbone,
        train_exp_logits_only=args.clf_train_exp_logits_only,
        train_mb_size=args.clf_batch_size,
        hat_reg_base_factor=args.clf_hat_reg_base_factor,
        num_replay_samples_per_batch=args.clf_num_replay_samples_per_batch,
        device=device,
        train_epochs=args.clf_num_epochs,
        logit_calibr=args.clf_logit_calibr,
        plugins=[*competition_plugins],
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
        rep.sync_replay_features(clf)

    print(f"Training done in {competition_plugins[0].time_spent:.2f} minutes.")

    # --- Make prediction on test-set samples
    # This is for testing purpose only.
    print("Testing with tta = 1 and tta = 18 ...")
    _final_acc = []
    for __tta in [1, 18]:
        tst_targets, tst_predictions, tst_logits, tst_features = clf.predict(
            benchmark.test_stream[0].dataset,
            tst_time_aug=__tta,  # Should be `args.tst_time_aug`
        )

        # Save predictions or print the results
        if args.benchmark:
            import dill

            with open("./data/challenge_test_labels.pkl", "rb") as __f:
                tst_labels = dill.load(__f)

            tst_labels = np.array(tst_labels)
            __acc = np.mean(tst_predictions == tst_labels)

            np.save(f"{args.name}_{args.config}_logits.npy", tst_logits)
            np.save(
                f"{args.name}_{args.config}_predictions.npy", tst_predictions
            )
        else:
            __acc = np.mean(tst_predictions == tst_targets)

        print(f"Test-set accuracy: {__acc}")
        _final_acc.append(__acc)

    # Report NNI result
    try:
        import nni

        nni.report_final_result(np.max(_final_acc))
    except ImportError:
        pass

    # Error analysis
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

# Everything below is trained with
# (1) linear-scaled decreasing regularization factor
# (2) number of replay sample of 8 in any batch
# (3) all the tricks from the post on how to accelerate training

# Config 1
# +-----------+-------------+-------------+----------------+-----------------+
# |           | rep_replay  | clf_replay  | clf_freeze_bb  | acc
# +-----------+-------------+-------------+----------------+-----------------+
# | train     | false       | false       | false          | 0.3068/0.3552
# +-----------+-------------+-------------+----------------+-----------------+
# | train(1)  | false       | true        | false          | 0.3116/0.3609
# +-----------+-------------+-------------+----------------+-----------------+
# | train(8)  | true        | true        | false          |
# +-----------+-------------+-------------+----------------+-----------------+
# | train(3)  | false       | true        | true           | 0.2969/0.3437
# +-----------+-------------+-------------+----------------+-----------------+
# | train(9)  | true        | true        | true           |
# +-----------+-------------+-------------+----------------+-----------------+


# Config 3
# +-----------+-------------+-------------+----------------+-----------------+
# |           | rep_replay  | clf_replay  | clf_freeze_bb  | acc
# +-----------+-------------+-------------+----------------+-----------------+
# | train(4)  | false       | false       | false          | 0.2361/0.2840
# +-----------+-------------+-------------+----------------+-----------------+
# | train(5)  | false       | true        | false          | 0.2519/0.3022
# +-----------+-------------+-------------+----------------+-----------------+
# | train(10) | true        | true        | false          |
# +-----------+-------------+-------------+----------------+-----------------+
# | train(7)  | false       | true        | true           | 0.2513/0.3020
# +-----------+-------------+-------------+----------------+-----------------+
# | train(11) | true        | true        | true           |
# +-----------+-------------+-------------+----------------+-----------------+


# On the topic of optimizers and learning rate
# Config 1
# +-----------------+----------+----------+-----------------+
# | optimizer       | rep_lr   | clr_lr   | acc             |
# +-----------------+----------+----------+-----------------+
# | Adam            | 0.01     | 0.001    | 0.3046/0.3598   |
# +-----------------+----------+----------+-----------------+
# | Adam            | 0.02     | 0.002    | 0.2818/0.3317   |
# +-----------------+----------+----------+-----------------+
# | Adam (amsgrad)  | 0.01     | 0.001    | 0.3073/0.3615   |
# +-----------------+----------+----------+-----------------+
# | Adam (amsgrad)  | 0.012    | 0.0012   | 0.3033/0.3563   |
# +-----------------+----------+----------+-----------------+
# | SGD (mmt=0.9)   | 0.01     | 0.001    | 0.2588/0.2873   |
# +-----------------+----------+----------+-----------------+
# | SGD (mmt=0.9)   | 0.02     | 0.002    | 0.2736/0.3148   |
# +-----------------+----------+----------+-----------------+
# | SGD (mmt=0.9)   | 0.04     | 0.004    | 0.2879/0.3323   |
# +-----------------+----------+----------+-----------------+
# | SGD (mmt=0.9)   | 0.08     | 0.008    | 0.2900/0.3359   |
# +-----------------+----------+----------+-----------------+
# | SGD (mmt=0.9)   | 0.10     | 0.010    | 0.2739/0.3221   |
# +-----------------+----------+----------+-----------------+
# | LARS (mmt=0.9)  | 0.01     | 0.001    | 0.2588/0.2873   |
# +-----------------+----------+----------+-----------------+
# | LARS (mmt=0.9)  | 0.02     | 0.002    | 0.2736/0.3148   |
# +-----------------+----------+----------+-----------------+
# | LARS (mmt=0.9)  | 0.04     | 0.004    | 0.2879/0.3323   |
# +-----------------+----------+----------+-----------------+

# On the topic of hat regularization factor
# Config 1
# 0 replay samples per batch for both rep and clf
# +----------------------------+-----------------+
# | rep_hat_reg_base_factor    | acc             |
# +----------------------------+-----------------+
# | 0.90                       |                 |  train
# +----------------------------+-----------------+
# | 0.85                       |                 |  train(1)
# +----------------------------+-----------------+
# | 0.80                       |                 |  train(2)
# +----------------------------+-----------------+
# | 0.75                       |                 |  train(3)
# +----------------------------+-----------------+
# | 0.70                       |                 |  train(4)
# +----------------------------+-----------------+
# | 0.65                       |                 |  train(5)
# +----------------------------+-----------------+

# TODO: check if the replay for reg is actually working

# TODO: make the reg curve a little less aggressive. Right now the masks are
#  almost fully utilized by task 45. Maybe we should also consider the
#  number of classes in the task when computing the reg factor.

