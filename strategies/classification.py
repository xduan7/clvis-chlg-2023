import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.templates import SupervisedTemplate
from hat import HATPayload
from hat.utils import get_hat_mask_scale, get_hat_reg_term, get_hat_util

# Validation set size (for temperature scaling)
val_ratio = 0.1

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
) -> torch.Tensor:
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
    _distances = torch.cdist(features, _centers)
    _nearest_indices = torch.argmin(_distances, dim=0)
    return features[_nearest_indices]


class Classification(SupervisedTemplate):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int,
        train_epochs: int,
        device: torch.device,
        freeze_backbone: bool,
        train_exp_logits_only: bool,
        logit_calibr: str,  # One of "none", "temp", or "norm"
        replay: bool,
        lr_decay: bool,
        plugins: Optional[List[SupervisedPlugin]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            plugins=plugins,
            evaluator=self._get_evaluator(verbose),
        )
        self.freeze_backbone = freeze_backbone
        self.train_exp_logits_only = train_exp_logits_only
        self.logit_calibr = logit_calibr
        self.hat_config = getattr(self.model, "hat_config", None)
        self.replay = replay
        self.lr_decay = lr_decay
        self.verbose = verbose

        if self.lr_decay:
            self.scheduler_kwargs = {
                "step_size": 10,
                "gamma": 0.1,
            }
            self.lr_scheduler_plugin = LRSchedulerPlugin(
                StepLR(self.optimizer, **self.scheduler_kwargs),
            )
            self.plugins.append(self.lr_scheduler_plugin)

        self.classes_in_experiences = []
        self.logit_norm = []
        self.logit_temp = nn.ParameterList([])
        self.replay_features = {}
        self.replay_feature_tensor = None
        self.replay_target_tensor = None
        self.num_replay_samples_per_class = 2
        self.num_replay_samples_per_batch = 2

        self.classes_trained_in_this_experience = None
        self.num_classes_trained_in_this_experience = None

    # TODO: different learning rates for backbone and head
    # TODO: rotated head
    # TODO: smooth labels
    # TODO: add weights to loss function if replay

    @staticmethod
    def _get_evaluator(verbose: bool):
        return EvaluationPlugin(
            accuracy_metrics(
                minibatch=False, epoch=True, experience=True, stream=True
            ),
            loss_metrics(
                minibatch=False, epoch=True, experience=True, stream=True
            ),
            forgetting_metrics(experience=True),
            loggers=[InteractiveLogger()] if verbose else None,
        )

    def _unpack_minibatch(self):
        self._check_minibatch()
        self.mbatch = (
            self.mbatch[0].to(self.device),
            self.mbatch[1].to(self.device),
            self.mbatch[2],
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
            __val_indexes = __indexes[: int(val_ratio * len(__indexes))]
            __trn_indexes = __indexes[int(val_ratio * len(__indexes)) :]
            __trn_dataset = __adapted_dataset.subset(__trn_indexes)
            __val_dataset = __adapted_dataset.subset(__val_indexes)
            self.adapted_dataset = __trn_dataset
            self.val_adapted_dataset = __val_dataset
            self.val_dataloader = DataLoader(
                self.val_adapted_dataset,
                batch_size=self.train_mb_size,
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
            persistent_workers=persistent_workers,
            collate_fn=_clf_collate_fn,
        )

    def model_adaptation(self, model=None):
        _model = super().model_adaptation(model)
        if self.freeze_backbone:
            for __p in _model.parameters():
                __p.requires_grad = False
            for __p in _model.linear.parameters():
                __p.requires_grad = True
        self.classes_in_experiences.append(
            self.experience.classes_in_this_experience
        )

        if self.replay and self.train_exp_logits_only:
            self.classes_trained_in_this_experience = (
                self.experience.classes_in_this_experience + [100]
            )
            # Reset the last neuron of the classifier head
            self.model.linear.weight.data[-1, :] = 0
            self.model.linear.bias.data[-1] = 0
        elif self.train_exp_logits_only:
            self.classes_trained_in_this_experience = (
                self.experience.classes_in_this_experience
            )
        else:
            self.classes_trained_in_this_experience = list(range(100))

        self.num_classes_trained_in_this_experience = len(
            self.classes_trained_in_this_experience
        )

        if self.logit_calibr == "norm":
            self.logit_norm.append(
                nn.BatchNorm1d(
                    self.num_classes_trained_in_this_experience,
                    affine=False,
                ).to(self.device)
            )
        elif self.logit_calibr == "temp":
            self.logit_temp.append(
                nn.Parameter(
                    torch.ones(self.num_classes_trained_in_this_experience),
                    requires_grad=True,
                ).to(self.device)
            )

        if self.replay:
            for __c in self.experience.classes_in_this_experience:
                if __c in self.replay_features:
                    del self.replay_features[__c]
            if len(self.replay_features) > 0:
                __logits, __y = [], []
                for __c, __l in self.replay_features.items():
                    # Shape of the logits are [num_samples, num_features]
                    __logits.append(__l)
                    if self.train_exp_logits_only:
                        __y += [
                            self.num_classes_trained_in_this_experience - 1
                        ] * len(__l)
                    else:
                        __y += [__c] * len(__l)
                self.replay_feature_tensor = torch.cat(__logits, dim=0)
                self.replay_target_tensor = torch.tensor(
                    __y, device=self.device
                )

        return _model

    def make_optimizer(self):
        # Recreate an optimizer with the new parameters to
        # (1) prevent momentum/gradient carry-over
        # (2) train on the new projection head
        self.optimizer = self.optimizer.__class__(
            self.model.parameters(),
            **self.optimizer.defaults,
        )
        if self.lr_decay:
            self.lr_scheduler_plugin.scheduler = StepLR(
                self.optimizer,
                **self.scheduler_kwargs,
            )

    def training_epoch(self, **kwargs):
        _num_mb = len(self.dataloader)
        for mb_it, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            if self.hat_config is None:
                _features = self.model.forward_features(self.mb_x)
            else:
                _task_id = self.experience.current_experience
                _progress = mb_it / (_num_mb - 1)
                _mask_scale = (
                    get_hat_mask_scale(
                        strat="cosine",
                        progress=_progress,
                        max_trn_mask_scale=self.hat_config.max_trn_mask_scale,
                    )
                    if not self.freeze_backbone
                    else self.hat_config.max_trn_mask_scale
                )
                _pld = HATPayload(
                    data=self.mb_x,
                    task_id=_task_id,
                    mask_scale=_mask_scale,
                )
                _features = self.model.forward_features(_pld)

            # TODO: insert logits from other classes here ... ?
            self.mb_output = self.model.forward_head(_features)[
                :, self.classes_trained_in_this_experience
            ]
            if self.logit_calibr == "norm":
                self.mb_output = self.logit_norm[
                    self.experience.current_experience
                ](self.mb_output)

            # Add replay samples here ..
            if (
                self.replay
                and self.replay_feature_tensor is not None
                # and (len(self.experience.classes_in_this_experience) <= 2)
            ):
                # TODO: Not sure how many replay samples to use here ...
                # Add the replay samples by forwarding the saved logits
                # Modify self.mb_output and self.mb_y accordingly
                __indices = torch.randperm(
                    self.replay_feature_tensor.shape[0]
                )[: self.num_replay_samples_per_batch]
                __replay_logits = self.model.forward_head(
                    self.replay_feature_tensor[__indices]
                )[:, self.classes_trained_in_this_experience]
                # Add random noise to the logits
                # __replay_logits = __replay_logits + torch.randn_like(
                #     __replay_logits
                # ) * 0.1
                __replay_targets = self.replay_target_tensor[__indices]
                self.mb_output = torch.cat(
                    [self.mb_output, __replay_logits], dim=0
                )
                self.mbatch = (
                    self.mbatch[0],
                    torch.cat([self.mb_y, __replay_targets], dim=0),
                    self.mbatch[2],
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()
            if self.hat_config is not None and not self.freeze_backbone:
                # noinspection PyUnboundLocalVariable
                self.loss = self.loss + get_hat_reg_term(
                    module=self.model,
                    strat="uniform",
                    task_id=_task_id,
                    mask_scale=_mask_scale,
                )
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        """Calibrate the logits with temperature scaling."""
        if self.logit_calibr == "temp":
            self.model.eval()
            _logits, _targets = [], []
            with torch.no_grad():
                for __images, __targets, __task_id in self.val_dataloader:
                    __images = __images.to(device=self.device)
                    __targets = __targets.to(device=self.device)

                    if self.hat_config is None:
                        __features = self.model.forward_features(__images)
                    else:
                        _pld = HATPayload(
                            data=__images,
                            task_id=__task_id,
                            mask_scale=self.hat_config.max_trn_mask_scale,
                        )
                        __features = self.model.forward_features(_pld)

                    __logits = self.model.forward_head(__features)
                    if self.train_exp_logits_only:
                        __logits = __logits[
                            :, self.experience.classes_in_this_experience
                        ]

                    _logits.append(__logits)
                    _targets.append(__targets)

            _logits = torch.cat(_logits).detach()
            _targets = torch.cat(_targets).detach()

            _temp = self.logit_temp[self.experience.current_experience]
            _optim = optim.LBFGS([_temp], lr=0.01, max_iter=50)

            def _eval_temperature():
                _optim.zero_grad()
                __loss = nn.CrossEntropyLoss()(_logits / _temp, _targets)
                __loss.backward()
                return __loss

            _optim.step(_eval_temperature)

        if self.replay:
            self.model.eval()

            # No augmentation for replay samples
            _dataset = self.adapted_dataset.replace_current_transform_group(
                (_tsfm, None)
            )
            _dataloader = DataLoader(
                _dataset,
                batch_size=self.train_mb_size * 2,
                shuffle=False,
                num_workers=8,
                pin_memory=(self.device.type == "cuda"),
                persistent_workers=False,
                collate_fn=_clf_collate_fn,
            )
            _features, _targets = [], []
            with torch.no_grad():
                for __images, __targets, _ in _dataloader:
                    __images = __images.to(device=self.device)
                    __targets = __targets.to(device=self.device)

                    if self.hat_config is None:
                        __features = self.model.forward_features(__images)
                    else:
                        _pld = HATPayload(
                            data=__images,
                            task_id=self.experience.current_experience,
                            mask_scale=self.hat_config.max_trn_mask_scale,
                        )
                        __features = self.model.forward_features(_pld)

                    _features.append(__features)
                    _targets.append(__targets)

            # Select two representative samples per class
            _features = torch.cat(_features).detach()
            _targets = torch.cat(_targets).detach()

            # _class_feature_means = []
            for __c in self.experience.classes_in_this_experience:
                __class_features = _features[_targets == __c]

                # Use K means to find representative samples
                self.replay_features[__c] = _k_means(
                    __class_features,
                    num_clusters=self.num_replay_samples_per_class,
                )

                # Select the closest samples to the class mean
                # __class_feature_mean = __class_features.mean(dim=0)
                # __distances = torch.norm(
                #     __class_features - __class_feature_mean, dim=1
                # )
                # __distances, __indices = torch.sort(__distances)
                # __indices = __indices[:self.num_replay_samples_per_class]
                # self.logit_replay[__c] = _features[__indices]

        # Unfreeze the model if needed
        if self.freeze_backbone:
            for __p in self.model.parameters():
                try:
                    __p.requires_grad = True
                except RuntimeError:
                    pass
        else:
            # Report the HAT mask usage
            if self.hat_config is not None and self.verbose:
                _mask_util_df = get_hat_util(module=self.model)
                print(_mask_util_df)

        # Call the plugin functions if there is any
        super()._after_training_exp(**kwargs)

    @torch.no_grad()
    def predict_by_all_exp(
        self,
        tst_dataset,
        tst_time_aug: int = 1,
        remove_extreme_logits: bool = False,
    ):
        """This is not an avalanche function."""
        assert self.train_exp_logits_only, (
            "This function is only for the case where the model is trained "
            "with `train_exp_logits_only=True`."
        )

        # Freeze the model including the batch norm layers.
        self.model.eval()
        for __n in self.logit_norm:
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

        # We need to get all the experiences that contain classes that are last
        # seen in the current experience.
        _cls_to_last_seen_exp = {}
        for __exp, __cls_in_exp in enumerate(self.classes_in_experiences):
            for __cls in __cls_in_exp:
                _cls_to_last_seen_exp[__cls] = __exp
        _last_seen_exp_to_cls = {}
        for __cld, __exp in _cls_to_last_seen_exp.items():
            if __exp not in _last_seen_exp_to_cls:
                _last_seen_exp_to_cls[__exp] = []
            _last_seen_exp_to_cls[__exp].append(__cld)

        _tst_features = torch.zeros(
            tst_time_aug,
            len(_tst_dataset),
            160,
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
        for __j in range(tst_time_aug):
            _start_idx = 0
            for __i, (__images, __targets, _) in enumerate(_tst_dataloader):
                __images = __images.to(self.device)
                __targets = __targets.to(self.device)
                __end_idx = _start_idx + len(__targets)

                for (
                    __exp_id,
                    __classes_in_exp,
                ) in _last_seen_exp_to_cls.items():
                    if self.hat_config is None:
                        __features = self.model.forward_features(__images)
                        __logits = self.model.forward_head(__features)
                    else:
                        __pld = HATPayload(
                            data=__images,
                            task_id=__exp_id,
                            mask_scale=self.hat_config.max_trn_mask_scale,
                        )
                        __features = self.model.forward_features(__pld)
                        __logits = self.model.forward_head(__features)
                    if self.logit_calibr == "norm":
                        __logits = __logits[:, __classes_in_exp]
                        __logits = self.logit_norm[__exp_id](__logits)
                    elif self.logit_calibr == "temp":
                        __all_classes_in_exp = self.classes_in_experiences[
                            __exp_id
                        ]
                        __logits[:, __all_classes_in_exp] = (
                            __logits[:, __all_classes_in_exp]
                            / self.logit_temp[__exp_id]
                        )
                        __logits = __logits[:, __classes_in_exp]
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
        for __n in self.logit_norm:
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
            160,
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
                    __logits = self.model.forward_head(__features)
                else:
                    __pld = HATPayload(
                        data=__images,
                        task_id=(self.hat_config.num_tasks - 1),
                        mask_scale=self.hat_config.max_trn_mask_scale,
                    )
                    __features = self.model.forward_features(__pld)
                    __logits = self.model.forward_head(__features)

                if self.logit_calibr == "norm":
                    __logits = __logits[:, self.classes_in_experiences[-1]]
                    __logits = self.logit_norm[-1](__logits)
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
        remove_extreme_logits: bool = False,
    ):
        if self.train_exp_logits_only:
            return self.predict_by_all_exp(
                tst_dataset,
                tst_time_aug,
                remove_extreme_logits,
            )
        else:
            return self.predict_by_last_exp(
                tst_dataset,
                tst_time_aug,
                remove_extreme_logits,
            )
