#######################################
#     DO NOT CHANGE THESE IMPORTS

import sys

sys.path.insert(0, "avalanche")

#######################################

import argparse
import json
import os

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
from utils.get_ckpt_dir_path import get_ckpt_dir_path
from utils.params import merge_params, print_params

# https://github.com/pytorch/pytorch/issues/8126
os.system("taskset -p 0xffffffffffffffffffffffffffffffff %d" % os.getpid())
# https://discuss.pytorch.org/t/cpu-cores-not-working/19222
torch.set_num_threads(20)
# torch.multiprocessing.set_start_method('spawn', force=True)


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
        choices=[1, 2, 3, 4, 5, 6],
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
        "--hat_reg_decay_exp",
        type=float,
        default=1.0,
        help="The decay exponent for the HAT regularization term over time. "
        "1.0 means the decay is linear (experience n has (49-n)/49 "
        "decay. 0.0 means no decay.",
    )
    parser.add_argument(
        "--hat_reg_enrich_ratio",
        type=float,
        default=0.0,
        help="The enrichment ratio of the HAT regularization term "
        "depending on the number of classes in a experience. 1.0 means the "
        "regularization term is increased by (c - 25)% where c is the "
        "number of classes in the current experience. 0.0 means no "
        "enrichment.",
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
        "--clf_freeze_hat",
        action="store_true",
        help="Freeze the HAT maskers during the classification learning phase.",
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
        default="none",
        choices=["none", "temp", "norm", "batchnorm"],
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
        hat_reg_decay_exp=args.hat_reg_decay_exp,
        hat_reg_enrich_ratio=args.hat_reg_enrich_ratio,
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
        freeze_hat=args.clf_freeze_hat,
        hat_reg_base_factor=args.clf_hat_reg_base_factor,
        hat_reg_decay_exp=args.hat_reg_decay_exp,
        hat_reg_enrich_ratio=args.hat_reg_enrich_ratio,
        num_replay_samples_per_batch=args.clf_num_replay_samples_per_batch,
        device=device,
        train_epochs=args.clf_num_epochs,
        logit_calibr=args.clf_logit_calibr,
        plugins=[*competition_plugins],
        verbose=args.verbose,
    )

    # --- Training Loops
    # Check if there is a checkpoint to load
    ckpt_dir_path = get_ckpt_dir_path(args)
    print(f"ckpt_dir_path: {ckpt_dir_path}")
    if clf.has_ckpt(ckpt_dir_path):
        clf.load_ckpt(ckpt_dir_path)
    else:
        # Save a readable copy of the args for reference (with json)
        with open(os.path.join(ckpt_dir_path, "args.json"), "w") as __f:
            json.dump(vars(args), __f, indent=4)

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

        print(
            f"Training done in {competition_plugins[0].time_spent:.2f} minutes."
        )
        clf.save_ckpt(ckpt_dir_path)

    # --- Make prediction on test-set samples
    # This is for testing purpose only.
    _final_acc = []

    _tst_dataset = benchmark.test_stream[0].dataset
    # Get a fraction of the test-set of size 32
    # _tst_dataset = benchmark.test_stream[0].dataset.subset(
    #     indices=np.random.choice(
    #         np.arange(len(benchmark.test_stream[0].dataset)),
    #         size=64,
    #         replace=False,
    #     )
    # )
    # tst_targets, tst_predictions, tst_logits, tst_features = clf.predict(
    #     _tst_dataset,
    #     tst_time_aug=args.tst_time_aug,
    #     num_exp=1,
    #     exp_trn_acc_lower_bound=None,
    #     ignore_singular_exp=None,
    #     remove_extreme_logits=True,
    # )
    _tst_config = [[1, 1], [18, 1], [18, 3]]
    for __tta, __num_exp in _tst_config:
        print(f"Testing with tta = {__tta} and num_exp = {__num_exp} ...")
        tst_targets, tst_predictions, tst_logits, tst_features = clf.predict(
            _tst_dataset,
            tst_time_aug=__tta,  # Should be `args.tst_time_aug`
            num_exp=__num_exp,
            exp_trn_acc_lower_bound=0.6,
            ignore_singular_exp=True,
            remove_extreme_logits=True,
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
        y_score=tst_logits,
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
# | 0.90                       | 0.3067/0.3603   |
# +----------------------------+-----------------+
# | 0.85                       | 0.3127/0.3646   |
# +----------------------------+-----------------+
# | 0.80                       | 0.3127/0.3736   |
# +----------------------------+-----------------+
# | 0.75                       | 0.3136/0.3585   |
# +----------------------------+-----------------+
# | 0.70                       | 0.3174/0.3602   |
# +----------------------------+-----------------+
# | 0.65                       | 0.3122/0.3605   |
# +----------------------------+-----------------+

# On the topic of number of replay samples per batch in clf
# Config 3
# +-------------------------------------+-----------------+
# | clf_num_replay_samples_per_batch    | acc             |
# +-------------------------------------+-----------------+
# | 0                                   | 0.2338/0.2773   |
# +-------------------------------------+-----------------+
# | 2                                   | 0.2448/0.2874   |
# +-------------------------------------+-----------------+
# | 4                                   | 0.2422/0.2888   |
# +-------------------------------------+-----------------+
# | 8                                   | 0.2431/0.2814   |
# +-------------------------------------+-----------------+
# | 12                                  | 0.2495/0.2914   |
# +-------------------------------------+-----------------+
# | 16                                  | 0.2346/0.2820   |
# +-------------------------------------+-----------------+
# | 20                                  | 0.2425/0.2855   |
# +-------------------------------------+-----------------+
# | 24                                  | 0.2265/0.2742   |
# +-------------------------------------+-----------------+
# | 32                                  | 0.2421/0.2892   |
# +-------------------------------------+-----------------+

# `train` is the rep with replay samples of 8 (working by drastically
# decreasing the hat reg factor). NOT WORKING.

# On the topic of number of epochs and training time
# Config 3 with 12 replay samples without enrichment factor.
# +-----------------+-----------------+
# | num_epochs      | acc             |
# +-----------------+-----------------+
# | 20              | 0.2335/0.2780   | train(1)
# +-----------------+-----------------+
# | 25              | 0.2428/0.2942   | train(2)
# +-----------------+-----------------+
# | 30              | 0.2436/0.2997   | train(3)
# +-----------------+-----------------+
# | 35              | 0.2396/0.2936   | train(4)
# +-----------------+-----------------+
# | 40              | 0.2217/0.2792   | train(5)
# +-----------------+-----------------+
# | 45              | 0.2290/0.2941   | train(6)
# +-----------------+-----------------+

# Another thing about more epochs is that, the mask regularization is more
# potent. I'm not sure how this impacts the performance of the model, but if
# I have to guess, it's probably not good.

# `train` is the rep default with **0.5 hat reg
# `train(1) is the rep with 1 hat reg
# `train(2) is the rep with 1 hat reg with frozen hat during clf
# `train(3) is the rep with 1 hat reg with drop_last=False

# TODO: different learning rate for clf backbone and clf head
# TODO: check the RAM and GPU memory usage (especially during inference)
# TODO: test the model with `torch.set_float32_matmul_precision('high')`

# TODO: this experiment needs a rerun
# `train` non reference 0.282/0.3164
# `train(1)` batch norm (the right way) 0.2728/0.3254
# `train(2)` temp 0.284/0.3272
# `train(3)` batch norm (the right way but not during training)
# `train(4)` norm 0.1652/0.1901
# `train(5)` norm (not in training) 0.2779/0.3222


