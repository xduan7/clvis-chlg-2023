from typing import List, Optional

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from hat import HATPayload
from hat.utils import get_hat_mask_scale, get_hat_reg_term


# Validation set size (for temperature scaling)
val_ratio = 0.1

# Original SupContrast augmentation without `RandomGrayscale`
_aug_tsfm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
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
        logit_calibr: str,  # "norm" or "temp"
        seed: int,
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
        self.seed = seed
        self.verbose = verbose

        self.classes_in_experiences = []
        self.logit_norm = []
        self.logit_temp = nn.ParameterList([])

    # TODO: initialization technique
    # TODO: logits normalization
    # TODO: rotated head
    # TODO: smooth labels
    # TODO: calibration (temperature scaling)
    # TODO: regularization for logits so that they are not too far from 0

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
            lambda __x: self.experience.classes_in_this_experience.index(__x)
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
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=_clf_collate_fn,
            )
        self.dataloader = DataLoader(
            self.adapted_dataset,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
        __num_classes = (
            len(self.experience.classes_in_this_experience)
            if self.train_exp_logits_only
            else 100
        )
        if self.logit_calibr == "norm":
            self.logit_norm.append(
                nn.BatchNorm1d(
                    __num_classes,
                    affine=False,
                ).to(self.device)
            )
        elif self.logit_calibr == "temp":
            self.logit_temp.append(
                nn.Parameter(
                    torch.ones(__num_classes),
                    requires_grad=True,
                ).to(self.device)
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
            self.mb_output = self.model.forward_head(_features)
            if self.train_exp_logits_only:
                self.mb_output = self.mb_output[
                    :, self.experience.classes_in_this_experience
                ]
            if self.logit_calibr == "norm":
                self.mb_output = self.logit_norm[
                    self.experience.current_experience
                ](self.mb_output)
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()
            if self.hat_config is not None and not self.freeze_backbone:
                # noinspection PyUnboundLocalVariable
                self.loss = self.loss + get_hat_reg_term(
                    module=self.model.base_model,
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

        # Unfreeze the model
        for __p in self.model.parameters():
            try:
                __p.requires_grad = True
            except RuntimeError:
                pass

    @torch.no_grad()
    def predict(self, tst_dataset):
        """This is not an avalanche function."""
        # Freeze the model including the batch norm layers.
        self.model.eval()
        for __n in self.logit_norm:
            __n.track_running_stats = False

        _tst_dataset = tst_dataset.replace_current_transform_group(
            (_tsfm, None)
        )
        _tst_dataloader = DataLoader(
            _tst_dataset,
            batch_size=512,
            num_workers=8,
            pin_memory=True,
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

        _tst_logits = torch.zeros(
            len(_tst_dataset),
            len(set(_tst_dataset.targets)),
            device=self.device,
        )
        _tst_logits[:] = torch.nan
        _tst_targets = -torch.ones(
            len(_tst_dataset),
            dtype=torch.long,
            device=self.device,
        )
        _start_idx = 0
        for __i, (__images, __targets, _) in enumerate(_tst_dataloader):
            __images = __images.to(self.device)
            __targets = __targets.to(self.device)
            __end_idx = _start_idx + len(__targets)

            for __exp_id, __classes_in_exp in _last_seen_exp_to_cls.items():
                if self.hat_config is None:
                    __logits = self.model.forward(__images)
                else:
                    __pld = HATPayload(
                        data=__images,
                        task_id=__exp_id,
                        mask_scale=self.hat_config.max_trn_mask_scale,
                    )
                    __logits = self.model.forward(__pld)
                if self.logit_calibr == "norm":
                    __logits = __logits[:, __classes_in_exp]
                    __logits = self.logit_norm[__exp_id](__logits)
                elif self.logit_calibr == "temp":
                    __all_classes_in_exp = self.classes_in_experiences[
                        __exp_id]
                    __logits[:, __all_classes_in_exp] = __logits[
                        :, __all_classes_in_exp
                    ] / self.logit_temp[__exp_id]
                    __logits = __logits[:, __classes_in_exp]
                assert torch.all(
                    torch.isnan(
                        _tst_logits[_start_idx:__end_idx, __classes_in_exp]
                    )
                )
                _tst_logits[_start_idx:__end_idx, __classes_in_exp] = __logits

            _tst_targets[_start_idx:__end_idx] = __targets
            _start_idx = __end_idx

        _tst_predictions = torch.argmax(_tst_logits, dim=1).cpu().numpy()
        _tst_logits = _tst_logits.cpu().numpy()
        _tst_targets = _tst_targets.cpu().numpy()
        return _tst_predictions, _tst_logits, _tst_targets
