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

# torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
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
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file.",
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
        "--hat_num_fragments",
        type=int,
        default=1,
        help="Number of fragments for the HAT model.",
    )
    parser.add_argument(
        "--hat_num_ensembles",
        type=int,
        default=1,
        help="Number of ensembles for the HAT model.",
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
        "--rep_proj_div_factor",
        type=float,
        default=1.0,
        help="The divergent factor for the embedding before the projected "
             "output of the samples and the replay embeddings."
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
        "--clf_logit_reg_factor",
        type=float,
        default=0.0,
        help="Regularization factor for replay logits during the "
             "classification learning phase.",
    )
    parser.add_argument(
        "--clf_logit_reg_degree",
        type=float,
        default=1.0,
        help="Degree of the regularization factor for replay logits "
             "during the classification learning phase.",
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
        "--clf_lr_scheduler",
        type=str,
        default="none",
        choices=["none", "multistep", "onecycle"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--tst_time_aug",
        type=int,
        default=1,
        help="Number of test-time augmentations. 1 means no augmentation.",
    )
    parser.add_argument(
        "--clf_use_momentum",
        action="store_true",
        help="Use momentum=3 with preset weights: 1, 2, 3 for classification",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
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
        scenario_config_file=args.config_file,
        seed=args.seed,
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
            nf=20,
            hat_config=hat_config,
            num_fragments=args.hat_num_fragments,
            num_ensembles=args.hat_num_ensembles,
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
        TimeChecker()
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
        proj_div_factor=args.rep_proj_div_factor,
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
        lr_scheduler=args.clf_lr_scheduler,
        freeze_backbone=args.clf_freeze_backbone,
        train_exp_logits_only=args.clf_train_exp_logits_only,
        train_mb_size=args.clf_batch_size,
        freeze_hat=args.clf_freeze_hat,
        hat_reg_base_factor=args.clf_hat_reg_base_factor,
        hat_reg_decay_exp=args.hat_reg_decay_exp,
        hat_reg_enrich_ratio=args.hat_reg_enrich_ratio,
        num_replay_samples_per_batch=args.clf_num_replay_samples_per_batch,
        logit_reg_factor=args.clf_logit_reg_factor,
        logit_reg_degree=args.clf_logit_reg_degree,
        device=device,
        train_epochs=args.clf_num_epochs,
        logit_calibr=args.clf_logit_calibr,
        plugins=[*competition_plugins],
        use_momentum=args.clf_use_momentum,
        verbose=args.verbose,
    )

    # --- Training Loops
    # Check if there is a checkpoint to load
    # ckpt_dir_path = get_ckpt_dir_path(args)
    # print(f"ckpt_dir_path: {ckpt_dir_path}")
    # if clf.has_ckpt(ckpt_dir_path):
    #     clf.load_ckpt(ckpt_dir_path)
    # else:
    #     # Save a readable copy of the args for reference (with json)
    #     with open(os.path.join(ckpt_dir_path, "args.json"), "w") as __f:
    #         json.dump(vars(args), __f, indent=4)

    for __exp in benchmark.train_stream:
        print(f"Training on experience {__exp.current_experience} ... ")
        if model.num_fragments > 1:
            model.copy_weights_from_previous_fragment(task_id=__exp.current_experience)

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
        # clf.save_ckpt(ckpt_dir_path)

    # --- Make prediction on test-set samples
    # This is for testing purpose only.
    _final_acc = []

    _tst_dataset = benchmark.test_stream[0].dataset
    tst_targets, tst_predictions, tst_logits, tst_features = clf.predict(
        _tst_dataset,
        tst_time_aug=args.tst_time_aug,
        num_exp=1,
        exp_trn_acc_lower_bound=None,
        ignore_singular_exp=None,
        remove_extreme_logits=True,
    )

    # Save predictions or print the results
    np.save(f"pred_{args.config_file.split('.')[0]}_{args.run_name}.npy", tst_predictions)

    import dill

    # with open("./data/challenge_test_labels.pkl", "rb") as __f:
    #     tst_labels = dill.load(__f)
    # tst_labels = np.array(tst_labels)

    tst_labels = np.load("./data/GT_test.npy")

    __acc = np.mean(tst_predictions == tst_labels)
    print(f"Accuracy: {__acc:.4f}")
    #np.save(f"{args.run_name}.npy", tst_predictions)


# Reference
# 1 * 1 -> 41.81%
# 1 * 2 -> 47.83%
# 10 * 1 -> 48.20%
# 50 * 2 -> 68.04% (< 500 minutes)

# train(1) is with 20 replay embeddings per class 43.64%
# train(2) is with 100 replay embeddings per class 42.70% K-means

# train(3) is with nf=40  49.36%
# train(4) is with nf=200 57.60% (1095.94 minutes)
# train(5) is with nf=400 58.97% (3293.63 minutes)

# train(6) is with old HAT 27.83% (normal mask init + linear mask scaling)

# train_ablation(1) is without TTA 37.05%
# train_ablation(2) is without momentum 39.39%
# train_ablation(3) is without TTA and momentum 33.78%

# train_ablation(4) is without logit calibration/normalization 37.72%

# TODO:
# no replay  43.05%
