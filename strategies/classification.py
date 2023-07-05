import math
import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from hat import HATPayload

from copy import deepcopy

from .base import BaseStrategy

# Validation set size (for temperature scaling)
temp_scale_val_ratio = 0.1
# Number of epochs for temperature scaling
temp_scale_num_epochs = 16
# Number of epochs for normalization scaling
norm_scale_num_epochs = 16

# Original SupContrast augmentation without `RandomGrayscale`
_aug_tsfm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ]
)
_tsfm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ]
)


def _clf_collate_fn(batch):
    _images, _targets, _task_ids = zip(*batch)
    _images = torch.stack(_images)
    _targets = torch.LongTensor(_targets)
    _task_id = _task_ids[0]
    return _images, _targets, _task_id


def _k_means(
    features: torch.Tensor,
    num_clusters: int,
    max_num_iter: int = 100,
    tol: float = 1e-4,
):
    _centers = features[torch.randperm(features.size(0))[:num_clusters]]
    for _ in range(max_num_iter):
        distances = torch.cdist(features, _centers)
        labels = torch.argmin(distances, dim=1)
        _new_centers = torch.stack(
            [features[labels == i].mean(dim=0) for i in range(num_clusters)]
        )
        if torch.norm(_new_centers - _centers) < tol:
            break
        _centers = _new_centers
    distances = torch.cdist(features, _centers)
    labels = torch.argmin(distances, dim=1)

    # If the labels only contain one cluster, then we need to re-run the
    # k-means algorithm
    # if len(torch.unique(labels)) == num_clusters:
    #     break
    # else:
    #     print("Re-running k-means algorithm ... ")

    # Get the features that are closest to each center
    _nearest_indices = []
    for __c in _centers:
        __dist = torch.cdist(features, __c.view(1, -1))
        _nearest_indices.append(torch.argmin(__dist, dim=0))
    return (
        features[_nearest_indices, :],
        torch.zeros_like(features[_nearest_indices, :]),
    )

    # Get the mean and std of each cluster
    # _mean, _std = [], []
    # for __i in torch.unique(labels):
    #     _mean.append(features[labels == __i, :].mean(dim=0))
    #     _std.append(features[labels == __i, :].std(dim=0))
    #     # If there is only one sample in the cluster, then set the std to 0
    #     if torch.isnan(_std[-1]).any():
    #         _std[-1] = torch.zeros_like(_std[-1])
    # return torch.stack(_mean), torch.stack(_std)


class CumulativeBatchNorm1d(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(dims))
        self.register_buffer('running_var', torch.zeros(dims))
        self.sample_count = 0

    def forward(self, x):
        batch_size = x.shape[0]
        new_sample_count = self.sample_count + batch_size
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, correction=0)
        if self.sample_count == 0:
            self.running_mean = mean
            self.running_var = var
        else:
            n1, n2 = self.sample_count, batch_size
            m1, m2 = self.running_mean, mean
            v1, v2 = self.running_var, var
            self.running_mean = mc = (n1*m1 + n2*m2) / (n1+n2)
            self.running_var = (n1*v1 + n2*v2 + n1*(m1-mc)**2 + n2*(m2-mc)**2) / (n1+n2)
        self.sample_count = new_sample_count
        return (x - self.running_mean) / (self.running_var + self.eps).sqrt()


class Classification(BaseStrategy):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: str,
        train_mb_size: int,
        train_epochs: int,
        hat_reg_base_factor: float,
        hat_reg_decay_exp: float,
        hat_reg_enrich_ratio: float,
        # num_replay_samples_per_batch: int,
        # logit_reg_factor: float,
        # logit_reg_degree: float,
        device: torch.device,
        freeze_hat: bool,
        freeze_backbone: bool,
        train_exp_logits_only: bool,
        logit_calibr: str,  # One of "none", "temp", or "norm"
        plugins: Optional[List[SupervisedPlugin]] = None,
        verbose: bool = True,
        use_momentum: bool = False,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            freeze_hat=freeze_hat,
            hat_reg_base_factor=hat_reg_base_factor,
            hat_reg_decay_exp=hat_reg_decay_exp,
            hat_reg_enrich_ratio=hat_reg_enrich_ratio,
            # num_replay_samples_per_batch=num_replay_samples_per_batch,
            device=device,
            plugins=plugins,
            verbose=verbose,
        )
        self.lr_scheduler_name = lr_scheduler
        self.freeze_backbone = freeze_backbone
        self.train_exp_logits_only = train_exp_logits_only
        self.logit_calibr = logit_calibr
        # self.logit_reg_factor = logit_reg_factor
        # self.logit_reg_degree = logit_reg_degree

        self.lr_scheduler = None
        self.classes_in_experiences = []
        self.classes_trained_in_experiences = []
        self.logit_norm = nn.ParameterList([])
        self.logit_temp = nn.ParameterList([])
        self.logit_batchnorm = nn.ModuleList([])
        self.logit_norm_clf = nn.ModuleList([])
        
        self.trn_acc = []

        self.classes_trained_in_this_experience = None
        self.num_classes_trained_in_this_experience = None

        self.use_momentum = use_momentum

        # preset 1
        # self.num_momentum = 4
        # self.cw = [2., 1., 3., 4]

        # preset 2
        # self.num_momentum = 4
        # self.cw = [1., 1., 3., 4]

        # preset 3
        self.num_momentum = 3
        self.cw = [1., 2., 3]

        # TODO: I don't need to copy weights here, I just need 3 copies of the linear head of the correct shape
        self.momentum_heads = nn.ModuleList([deepcopy(self.model.linear) for _ in range(self.num_momentum)])

    # TODO: different learning rates for backbone and head
    # TODO: rotated head
    # TODO: add weights to loss function if replay?

    @staticmethod
    def _get_criterion():
        return nn.CrossEntropyLoss()

    @staticmethod
    def _get_evaluator(verbose: bool):
        return EvaluationPlugin(
            accuracy_metrics(
                minibatch=False, epoch=True, experience=True, stream=True
            ),
            loss_metrics(
                minibatch=False, epoch=True, experience=True, stream=True
            ),
            # forgetting_metrics(experience=True),
            loggers=[InteractiveLogger()] if verbose else None,
        )

    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        _trgt_tsfm = (
            (lambda __x: self.experience.classes_in_this_experience.index(__x))
            if self.train_exp_logits_only
            else None
        )
        self.adapted_dataset = (
            self.adapted_dataset.replace_current_transform_group(
                (_aug_tsfm, _trgt_tsfm)
            )
        )

    # noinspection PyAttributeOutsideInit
    def make_train_dataloader(
        self,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        **kwargs,
    ):
        # Temperature scaling requires a validation set
        if self.logit_calibr == "temp":
            __adapted_dataset = self.adapted_dataset
            # Randomly choose 5% of the training set as validation set.
            __indexes = torch.randperm(len(__adapted_dataset))
            __val_indexes = __indexes[
                : int(temp_scale_val_ratio * len(__indexes))
            ]
            __trn_indexes = __indexes[
                int(temp_scale_val_ratio * len(__indexes)) :
            ]
            __trn_dataset = __adapted_dataset.subset(__trn_indexes)
            __val_dataset = __adapted_dataset.subset(__val_indexes)
            self.adapted_dataset = __trn_dataset
            self.val_adapted_dataset = __val_dataset
            self.val_dataloader = DataLoader(
                self.val_adapted_dataset,
                batch_size=self.train_mb_size * 2,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory if self.device.type == "cuda" else False,
                persistent_workers=persistent_workers,
                collate_fn=_clf_collate_fn,
            )
        self.dataloader = DataLoader(
            self.adapted_dataset,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory if self.device.type == "cuda" else False,
            pin_memory_device=str(self.device),
            persistent_workers=persistent_workers,
            collate_fn=_clf_collate_fn,
            drop_last=True,
        )

    def model_adaptation(self, model=None):
        _model = super().model_adaptation(model)
        if self.freeze_backbone:
            for __p in _model.parameters():
                __p.requires_grad = False
            for __p in _model.linear.parameters():
                __p.requires_grad = True

        self.classes_in_experiences.append(
            # sorted(self.experience.classes_in_this_experience)
            self.experience.classes_in_this_experience
        )

        # if self.replay and self.train_exp_logits_only:
        #     self.classes_trained_in_this_experience = (
        #         self.classes_in_experiences[-1] + [100]
        #     )
        #     # Reset the last neuron of the classifier head
        #     __bound = 1 / math.sqrt(_model.linear.weight.size(1))
        #     self.model.linear.weight.data[-1, :].uniform_(-__bound, __bound)
        #     self.model.linear.bias.data[-1] = 0
        # elif self.train_exp_logits_only:
        if self.train_exp_logits_only:
            self.classes_trained_in_this_experience = \
                self.classes_in_experiences[-1]
        else:
            self.classes_trained_in_this_experience = list(range(100))

        self.classes_trained_in_experiences.append(
            self.classes_trained_in_this_experience
        )
        self.num_classes_trained_in_this_experience = len(
            self.classes_trained_in_this_experience
        )

        if self.logit_calibr == "norm":
            # logit_norm is not learnable
            # contains the mean and std of the logits of the classes
            # trained in the current experience
            if len(self.logit_norm) == 0:
                _out_features = self.model.linear.weight.shape[0]
                # Add mean and std for the logits of the classes
                self.logit_norm.append(
                    nn.Parameter(
                        torch.zeros(_out_features),
                        requires_grad=False,
                    ).to(self.device, non_blocking=True)
                )
                self.logit_norm.append(
                    nn.Parameter(
                        torch.ones(_out_features),
                        requires_grad=False,
                    ).to(self.device, non_blocking=True)
                )
            self.logit_norm_clf.append(
                CumulativeBatchNorm1d(
                    self.num_classes_trained_in_this_experience,
                ).to(self.device, non_blocking=True)
            )
        elif self.logit_calibr == "temp":
            self.logit_temp.append(
                nn.Parameter(
                    torch.ones(self.num_classes_trained_in_this_experience),
                    requires_grad=True,
                ).to(self.device, non_blocking=True)
            )
        elif self.logit_calibr == "batchnorm":
            self.logit_batchnorm.append(
                nn.BatchNorm1d(
                    self.num_classes_trained_in_this_experience,
                    affine=False,
                ).to(self.device, non_blocking=True)
            )
            self.logit_norm_clf.append(
                CumulativeBatchNorm1d(
                    self.num_classes_trained_in_this_experience,
                ).to(self.device, non_blocking=True)
            )
        # self._del_replay_features(self.classes_in_experiences[-1])
        # if self.train_exp_logits_only:
        #     self._construct_replay_tensors(
        #         target=self.num_classes_trained_in_this_experience - 1
        #     )
        # else:
        #     self._construct_replay_tensors()
        return _model

    def make_optimizer(self):
        super().make_optimizer()
        if self.lr_scheduler_name == "multistep":
            _milestones = [
                math.floor(self.train_epochs * 0.8),
                math.floor(self.train_epochs * 0.9),
            ]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=_milestones,
                gamma=0.1,
            )
        elif self.lr_scheduler_name == "onecycle":
            _lr = self.optimizer.param_groups[0]["lr"]
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=_lr * 10.0,
                epochs=self.train_epochs,
                steps_per_epoch=len(self.dataloader),
            )
        elif self.lr_scheduler_name == "none":
            self.lr_scheduler = None
        else:
            raise ValueError(f"Invalid lr_scheduler value {self.lr_scheduler}")

    def training_epoch(self, **kwargs):
        _num_mb = len(self.dataloader)
        for mb_it, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            # self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad(set_to_none=True)
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            _features = self.forward_(
                model=self.model,
                images=self.mb_x,
                mb_it=mb_it,
                return_features=True,
            )
            _logits = self.model.forward_head(_features)
            self.mb_output = _logits[
                :, self.classes_trained_in_this_experience
            ]
            if self.logit_calibr == "batchnorm":
                self.mb_output = self.logit_batchnorm[self.task_id](self.mb_output)
            elif self.logit_calibr == "norm":
                __mean = self.logit_norm[0][self.classes_trained_in_this_experience]
                __std = self.logit_norm[1][self.classes_trained_in_this_experience]
                self.mb_output = (self.mb_output - __mean) / __std

            # Add replay samples here ..
            # __replay_features, __replay_targets, __replay_logits = \
            #     self._get_replay_samples()
            # if __replay_features is not None:
            #     __logits = self.model.forward_head(
            #         __replay_features)
            #     __trained_logits = __logits[
            #         :, self.classes_trained_in_this_experience
            #     ]
            #     self.mb_output = torch.cat(
            #         [self.mb_output, __trained_logits], dim=0
            #     )
            #     self.mbatch = (
            #         self.mbatch[0],
            #         torch.cat([self.mb_y, __replay_targets], dim=0),
            #         self.mbatch[2],
            #     )
            #     self.loss += self.logit_reg_factor * torch.dist(
            #         __logits[:100],
            #         __replay_logits[:100],
            #         p=self.logit_reg_degree,
            #     )
            #     # Add pair wise distances of embeddings between the replay
            #     # samples and the current samples to the loss
            #     # self.loss += self.emb_div_factor * torch.cdist(
            #     #     __replay_features,
            #     #     _features,
            #     #     p=self.emb_div_degree,
            #     # ).mean()

            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)
        if self.lr_scheduler_name == "onecycle":
            self.lr_scheduler.step()

    def _after_training_epoch(self, **kwargs):
        super()._after_training_epoch(**kwargs)
        if self.lr_scheduler_name == "multistep":
            self.lr_scheduler.step()

    def _after_training_exp(self, **kwargs):
        """Calibrate the logits with temperature scaling."""
        if self.use_momentum:
            # update the momentum weights
            cls_in_exp = self.classes_trained_in_this_experience
            for i in range(self.num_momentum-1, 0, -1):
                # weights of the linear is transposed, so the indexing is like this
                self.momentum_heads[i].weight.data[cls_in_exp, :] = deepcopy(self.momentum_heads[i-1].weight.data[cls_in_exp, :])
                self.momentum_heads[i].bias.data[cls_in_exp] = deepcopy(self.momentum_heads[i-1].bias.data[cls_in_exp])
            self.momentum_heads[0].weight.data[cls_in_exp, :] = deepcopy(self.model.linear.weight.data[cls_in_exp, :])
            self.momentum_heads[0].bias.data[cls_in_exp] = deepcopy(self.model.linear.bias.data[cls_in_exp])

        if self.logit_calibr == "temp":
            self._train_logit_temp(num_epochs=temp_scale_num_epochs)
        elif self.logit_calibr == "norm":
            self._calculate_logit_norm(num_epochs=norm_scale_num_epochs)

        # self._collect_replay_samples()
        # Record the accuracy on the training set (in a hacky way)
        self.trn_acc.append(self.plugins[1].metrics[0]._metric.result())
        super()._after_training_exp(**kwargs)

    # def _collect_replay_samples(self):
    #     self.model.eval()
    #     # No augmentation for replay samples
    #     _dataset = self.adapted_dataset.replace_current_transform_group(
    #         (_tsfm, None)
    #     )
    #     _dataloader = DataLoader(
    #         _dataset,
    #         batch_size=self.train_mb_size * 2,
    #         shuffle=False,
    #         num_workers=8,
    #         pin_memory=(self.device.type == "cuda"),
    #         persistent_workers=False,
    #         collate_fn=_clf_collate_fn,
    #     )
    #
    #     _features, _targets = [], []
    #     with torch.no_grad():
    #         for __images, __targets, _ in _dataloader:
    #             __images = __images.to(device=self.device, non_blocking=True)
    #             __targets = __targets.to(device=self.device, non_blocking=True)
    #             __features = self.forward_(
    #                 model=self.model,
    #                 images=__images,
    #                 return_features=True,
    #                 mask_scale=self.hat_config.max_trn_mask_scale,
    #             )
    #             _features.append(__features)
    #             _targets.append(__targets)
    #
    #     # Select two representative samples per class
    #     _features = torch.cat(_features).detach()
    #     _targets = torch.cat(_targets).detach()
    #
    #     # _class_feature_means = []
    #     for __c in self.classes_in_experiences[-1]:
    #         __class_features = _features[_targets == __c]
    #
    #         # Use K means to find representative samples
    #         self.replay_features[__c] = _k_means(
    #             __class_features,
    #             num_clusters=self.num_replay_samples_per_class,
    #         )

    def _train_logit_temp(self, num_epochs=1):
        self.model.eval()
        _logits, _targets = [], []
        for __e in range(num_epochs):
            with torch.no_grad():
                for __images, __targets, __task_id in self.val_dataloader:
                    __images = __images.to(device=self.device)
                    __targets = __targets.to(device=self.device)

                    __logits = self.forward_(
                        model=self.model,
                        images=__images,
                        return_features=False,
                        mask_scale=self.hat_config.max_trn_mask_scale,
                    ).detach()
                    # if self.hat_config is None:
                    #     __features = self.model.forward_features(__images)
                    # else:
                    #     _pld = HATPayload(
                    #         data=__images,
                    #         task_id=__task_id,
                    #         mask_scale=self.hat_config.max_trn_mask_scale,
                    #     )
                    #     __features = self.model.forward_features(_pld)
                    #
                    # __logits = self.model.forward_head(__features)
                    __logits = __logits[
                               :, self.classes_trained_in_this_experience
                               ]
                    _logits.append(__logits)
                    _targets.append(__targets)

        _logits = torch.cat(_logits)
        _targets = torch.cat(_targets)

        _temp = self.logit_temp[self.task_id]
        _optim = optim.LBFGS([_temp], lr=0.01, max_iter=50)

        def _eval_temperature():
            _optim.zero_grad()
            __loss = nn.CrossEntropyLoss()(_logits / _temp, _targets)
            __loss.backward()
            return __loss

        _optim.step(_eval_temperature)

    @torch.no_grad()
    def _calculate_logit_norm(self, num_epochs=1):
        self.model.eval()
        _logits = []
        for __e in range(num_epochs):
            for __images, __targets, __task_id in self.dataloader:
                __images = __images.to(device=self.device)
                __targets = __targets.to(device=self.device)
                __logits = self.forward_(
                    model=self.model,
                    images=__images,
                    return_features=False,
                    mask_scale=self.hat_config.max_trn_mask_scale,
                ).detach()
                # if self.hat_config is None:
                #     __features = self.model.forward_features(__images)
                # else:
                #     _pld = HATPayload(
                #         data=__images,
                #         task_id=__task_id,
                #         mask_scale=self.hat_config.max_trn_mask_scale,
                #     )
                #     __features = self.model.forward_features(_pld)
                #
                # __logits = self.model.forward_head(__features)
                __logits = __logits[
                           :, self.classes_trained_in_this_experience
                           ]
                _logits.append(__logits)

        _logits = torch.cat(_logits)
        self.logit_norm[0][self.classes_trained_in_this_experience] = _logits.mean(dim=0)
        self.logit_norm[1][self.classes_trained_in_this_experience] = _logits.std(dim=0)

    @torch.no_grad()
    def predict_by_all_exp(
        self,
        tst_dataset,
        tst_time_aug: int = 1,
        num_exp: int = 1,
        exp_weights: Optional[Sequence[float]] = None,
        exp_trn_acc_lower_bound: Optional[Sequence[float]] = None,
        ignore_singular_exp: bool = False,
        remove_extreme_logits: bool = True,
    ):
        """This is not an avalanche function."""
        assert self.train_exp_logits_only, (
            "This function is only for the case where the model is trained "
            "with `train_exp_logits_only=True`."
        )

        self.model.eval()

        _tst_dataset = tst_dataset.replace_current_transform_group(
            (_tsfm if tst_time_aug == 1 else _aug_tsfm, None)
        )
        _tst_dataloader = DataLoader(
            _tst_dataset,
            batch_size=1000,
            num_workers=8,
            pin_memory=True if self.device.type == "cuda" else False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=_clf_collate_fn,
        )

        # We need to get all the experiences that contain classes that are last
        # seen in the current experience.
        _cls_to_last_seen_exp = {}
        # TODO: remove this hardcoded 100
        n_classes = 100
        n_exps = len(self.classes_in_experiences)
        _cls_to_all_last_seen_exp = {i:[] for i in range(n_classes)}
        num_classes_per_exp = [len(i) for i in self.classes_in_experiences]
        inv_plus_5 = [1/(i+5) for i in num_classes_per_exp]
        count_factor = torch.tensor(inv_plus_5, device=self.device)
        count_weights = torch.tensor(self.cw, device=self.device)

        for __exp, __cls_in_exp in enumerate(self.classes_in_experiences):
            for __cls in __cls_in_exp:
                _cls_to_last_seen_exp[__cls] = __exp
                _cls_to_all_last_seen_exp[__cls].append(__exp)
        _last_seen_exp_to_cls = {}
        for __cld, __exp in _cls_to_last_seen_exp.items():
            if __exp not in _last_seen_exp_to_cls:
                _last_seen_exp_to_cls[__exp] = []
            _last_seen_exp_to_cls[__exp].append(__cld)

        _exp_to_n_last_seen = {exp:{} for exp in range(n_exps)} # exp: {history_step: cls}
        for cls, exps in _cls_to_all_last_seen_exp.items():
            exps = reversed(exps[-self.num_momentum:])
            for i, exp in enumerate(exps):
                if not i in _exp_to_n_last_seen[exp]:
                    _exp_to_n_last_seen[exp][i] = []
                _exp_to_n_last_seen[exp][i].append(cls)

        _tst_features = torch.zeros(
            tst_time_aug,
            len(_tst_dataset),
            self.model.linear.in_features,
            device=self.device,
        )
        _tst_logits = torch.zeros(
            tst_time_aug,
            len(_tst_dataset),
            len(_cls_to_last_seen_exp),
            device=self.device,
        )
        _tst_logits[:] = torch.nan
        _tst_targets = -torch.ones(
            len(_tst_dataset),
            dtype=torch.long,
            device=self.device,
        )

        # layer per exp per momentum step, so data is passed through each layer once
        _cbn_layers = {}
        for __j in range(tst_time_aug):
            _start_idx = 0
            for __i, (__images, __targets, _) in enumerate(_tst_dataloader):
                __images = __images.to(self.device)
                __targets = __targets.to(self.device)
                __end_idx = _start_idx + len(__targets)


                if self.use_momentum:
                    _tmp_test_logits_momentum = torch.zeros(
                        self.num_momentum,
                        len(__targets),
                        n_classes,
                        device=self.device,
                    )
                    
                    for __exp_id, __classes_in_exp in _exp_to_n_last_seen.items():
                        if self.hat_config is None:
                            __features = self.model.forward_features(__images)
                        else:
                            __pld = HATPayload(
                                data=__images,
                                task_id=__exp_id,
                                mask_scale=self.hat_config.max_trn_mask_scale,
                            )
                            __features = self.model.forward_features(__pld)
                        for i in range(self.num_momentum):
                            _cls = list(sorted(__classes_in_exp.get(i, [])))
                            if len(_cls) > 0:
                                if (__exp_id, i) not in _cbn_layers:
                                    _cbn_layers[(__exp_id, i)] = CumulativeBatchNorm1d(len(_cls)).to(self.device) if self.logit_calibr == "batchnorm" else nn.Identity()
                                _logit_norm_clf = _cbn_layers[(__exp_id, i)]

                                __logits = self.momentum_heads[i](__features)
                                # we must pass all classes in this exp as defined in training to the CBN layer, otherwise the positions are wrong
                                __logits = __logits[:, _cls]
                                __logits = _logit_norm_clf(__logits)

                                _tmp_test_logits_momentum[i, :, _cls] = __logits

                    for class_id, exp_list in _cls_to_all_last_seen_exp.items():
                        exp_list = exp_list[-self.num_momentum:]
                        available_len = len(exp_list)
                        _count_weights = count_weights[-available_len:]
                        _weights = _count_weights * count_factor[exp_list]
                        _weights = _weights / _weights.sum()
                        weights = torch.flip(_weights, (0,))
                        logits_with_momentum = _tmp_test_logits_momentum[:available_len, :, class_id]
                        _tst_logits[__j, _start_idx:__end_idx, class_id] = weights @ logits_with_momentum

                    _tst_targets[_start_idx:__end_idx] = __targets
                    _start_idx = __end_idx

                else:
                    for (
                        __exp_id,
                        __classes_in_exp,
                    ) in _last_seen_exp_to_cls.items():
                        if self.hat_config is None:
                            __features = self.model.forward_features(__images)
                        else:
                            __pld = HATPayload(
                                data=__images,
                                task_id=__exp_id,
                                mask_scale=self.hat_config.max_trn_mask_scale,
                            )
                            __features = self.model.forward_features(__pld)
                        __logits = self.model.forward_head(__features)
                        __logits = __logits[:, __classes_in_exp]
                        __logits = self.logit_norm_clf[__exp_id](__logits)
                        # Make sure that the logits for all filled with no overlap
                        # with the logits of other classes.
                        assert torch.all(
                            torch.isnan(
                                _tst_logits[
                                    __j, _start_idx:__end_idx, __classes_in_exp
                                ]
                            )
                        )
                        _tst_logits[
                            __j, _start_idx:__end_idx, __classes_in_exp
                        ] = __logits
                        _tst_features[__j, _start_idx:__end_idx] = __features
                    _tst_targets[_start_idx:__end_idx] = __targets
                    _start_idx = __end_idx

        if tst_time_aug > 3 and remove_extreme_logits:
            # For each sample, remove the highest and lowest logits
            _tst_logits_ = _tst_logits.clone()
            _tst_logits_ = _tst_logits_.sort(dim=0).values
            _tst_logits_ = _tst_logits_[1:-1]
            _tst_predictions = (
                torch.argmax(
                    _tst_logits_.mean(dim=0),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        else:
            _tst_predictions = (
                torch.argmax(
                    _tst_logits.mean(dim=0),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        _tst_logits = _tst_logits.cpu().numpy()
        _tst_targets = _tst_targets.cpu().numpy()
        _tst_features = _tst_features.cpu().numpy()
        return _tst_targets, _tst_predictions, _tst_logits, _tst_features

    @torch.no_grad()
    def predict_by_last_exp(
        self,
        tst_dataset,
        tst_time_aug: int = 1,
        remove_extreme_logits: bool = False,
    ):
        """This is not an avalanche function."""
        assert not self.train_exp_logits_only, (
            "This function is only for the case where the model is trained "
            "with `train_exp_logits_only=False`."
        )

        # Freeze the model including the batch norm layers.
        self.model.eval()
        for __n in self.logit_batchnorm:
            __n.track_running_stats = False

        _tst_dataset = tst_dataset.replace_current_transform_group(
            (_tsfm if tst_time_aug == 1 else _aug_tsfm, None)
        )
        _tst_dataloader = DataLoader(
            _tst_dataset,
            batch_size=512,
            num_workers=8,
            pin_memory=True if self.device.type == "cuda" else False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=_clf_collate_fn,
        )

        _tst_features = torch.zeros(
            tst_time_aug,
            len(_tst_dataset),
            self.model.linear.in_features,
            device=self.device,
        )
        _tst_logits = torch.zeros(
            tst_time_aug,
            len(_tst_dataset),
            100,
            device=self.device,
        )
        _tst_targets = -torch.ones(
            len(_tst_dataset),
            dtype=torch.long,
            device=self.device,
        )

        for __j in range(tst_time_aug):
            _start_idx = 0
            for __i, (__images, __targets, _) in enumerate(_tst_dataloader):
                __images = __images.to(self.device)
                __targets = __targets.to(self.device)
                __end_idx = _start_idx + len(__targets)

                if self.hat_config is None:
                    __features = self.model.forward_features(__images)
                else:
                    __pld = HATPayload(
                        data=__images,
                        task_id=(self.hat_config.num_tasks - 1),
                        mask_scale=self.hat_config.max_trn_mask_scale,
                    )
                    __features = self.model.forward_features(__pld)
                __logits = self.model.forward_head(__features)

                if self.logit_calibr == "batchnorm":
                    __logits = __logits[:, self.classes_in_experiences[-1]]
                    __logits = self.logit_batchnorm[-1](__logits)
                elif self.logit_calibr == "temp":
                    __logits = __logits[:, self.classes_in_experiences[-1]]
                    __logits = __logits / self.logit_temp[-1]

                _tst_logits[__j, _start_idx:__end_idx] = __logits
                _tst_features[__j, _start_idx:__end_idx] = __features
                _tst_targets[_start_idx:__end_idx] = __targets
                _start_idx = __end_idx

        if tst_time_aug > 3 and remove_extreme_logits:
            # For each sample, remove the highest and lowest logits
            _tst_logits_ = _tst_logits.clone()
            _tst_logits_ = _tst_logits_.sort(dim=0).values
            _tst_logits_ = _tst_logits_[1:-1]
            _tst_predictions = (
                torch.argmax(
                    _tst_logits_.mean(dim=0),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        else:
            _tst_predictions = (
                torch.argmax(
                    _tst_logits.mean(dim=0),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
        _tst_logits = _tst_logits.cpu().numpy()
        _tst_targets = _tst_targets.cpu().numpy()
        _tst_features = _tst_features.cpu().numpy()
        return _tst_targets, _tst_predictions, _tst_logits, _tst_features

    def predict(
        self,
        tst_dataset,
        tst_time_aug: int = 1,
        remove_extreme_logits: bool = True,
        **kwargs,
    ):
        if self.train_exp_logits_only:
            return self.predict_by_all_exp(
                tst_dataset=tst_dataset,
                tst_time_aug=tst_time_aug,
                remove_extreme_logits=remove_extreme_logits,
                **kwargs,
            )
        else:
            return self.predict_by_last_exp(
                tst_dataset=tst_dataset,
                tst_time_aug=tst_time_aug,
                remove_extreme_logits=remove_extreme_logits,
                **kwargs,
            )

    @staticmethod
    def has_ckpt(ckpt_dir_path: str):
        return os.path.exists(os.path.join(ckpt_dir_path, "clf_model.pth"))

    def save_ckpt(self, ckpt_dir_path: str):
        torch.save(self.model, os.path.join(ckpt_dir_path, "clf_model.pth"))
        torch.save(
            self.logit_norm,
            os.path.join(ckpt_dir_path, "clf_logit_norm.pth"),
        )
        torch.save(
            self.logit_temp,
            os.path.join(ckpt_dir_path, "clf_logit_temp.pth")
        )
        torch.save(
            self.logit_batchnorm,
            os.path.join(ckpt_dir_path, "clf_logit_batchnorm.pth")
        )
        torch.save(
            self.trn_acc, os.path.join(ckpt_dir_path, "clf_trn_acc.pth")
        )
        torch.save(
            self.classes_in_experiences,
            os.path.join(ckpt_dir_path, "clf_classes_in_experiences.pth"),
        )
        torch.save(
            self.classes_trained_in_experiences,
            os.path.join(
                ckpt_dir_path, "clf_classes_trained_in_experiences.pth"
            ),
        )
        # torch.save(
        #     self.replay_features,
        #     os.path.join(ckpt_dir_path, "clf_replay_features.pth"),
        # )

    def load_ckpt(self, ckpt_dir_path: str):
        self.model = torch.load(
            os.path.join(ckpt_dir_path, "clf_model.pth")
        ).to(self.device)
        self.logit_norm = torch.load(
            os.path.join(ckpt_dir_path, "clf_logit_norm.pth")
        ).to(self.device)
        self.logit_temp = torch.load(
            os.path.join(ckpt_dir_path, "clf_logit_temp.pth")
        ).to(self.device)
        self.logit_batchnorm = torch.load(
            os.path.join(ckpt_dir_path, "clf_logit_batchnorm.pth")
        ).to(self.device)
        self.trn_acc = torch.load(
            os.path.join(ckpt_dir_path, "clf_trn_acc.pth")
        )
        self.classes_in_experiences = torch.load(
            os.path.join(ckpt_dir_path, "clf_classes_in_experiences.pth")
        )
        self.classes_trained_in_experiences = torch.load(
            os.path.join(
                ckpt_dir_path, "clf_classes_trained_in_experiences.pth"
            )
        )
        # _replay_features = torch.load(
        #     os.path.join(ckpt_dir_path, "clf_replay_features.pth")
        # )
        # for __c, (__mean, __std) in _replay_features.items():
        #     self.replay_features[__c] = (
        #         __mean.to(self.device),
        #         __std.to(self.device)
        #     )
