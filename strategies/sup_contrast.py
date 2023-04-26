from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from hat import HATPayload
from hat.utils import get_hat_mask_scale, get_hat_reg_term, get_hat_util

# Original SupContrast augmentation
_aug_tsfm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        # TODO: should I add rotation here as well?
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ]
)


def _sup_contrast_collate_fn(batch):
    _images, _targets, _task_ids = zip(*batch)
    _images_i, _images_j = [], []
    for __image in _images:
        _images_i.append(_aug_tsfm(__image))
        _images_j.append(_aug_tsfm(__image))
    _images_i = torch.stack(_images_i)
    _images_j = torch.stack(_images_j)
    _images = torch.cat([_images_i, _images_j], dim=0)
    _targets = torch.LongTensor(_targets)
    _task_id = _task_ids[0]
    return _images, _targets, _task_id


class SupContrastLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    Author: Yonglong Tian (yonglong@mit.edu)
    """

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
    ):
        super(SupContrastLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if
                sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        device = features.device
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    "Num of labels does not match num of features"
                )
            mask = torch.eq(labels, labels.T).float().to(device)
        elif labels is None and mask is not None:
            mask = mask.float().to(device)

        # Features must be normalized similar to `NTXentLoss`
        features = nn.functional.normalize(features, dim=-1, eps=self.eps)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True) + self.eps
        )

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupContrast(SupervisedTemplate):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int,
        train_epochs: int,
        device: torch.device,
        proj_head_dim: int,
        plugins: Optional[List[SupervisedPlugin]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            criterion=SupContrastLoss(),
            device=device,
            plugins=plugins,
            evaluator=self._get_evaluator(verbose),
        )
        self.proj_head_dim = proj_head_dim
        self.hat_config = getattr(model, "hat_config", None)
        self.verbose = verbose

    # TODO: rotated head

    @staticmethod
    def _get_evaluator(verbose: bool):
        return EvaluationPlugin(
            loss_metrics(
                minibatch=True, epoch=True, experience=True, stream=True
            ),
            loggers=[InteractiveLogger()] if verbose else None,
        )

    def _unpack_minibatch(self):
        self._check_minibatch()
        self.mbatch = (
            self.mbatch[0].to(self.device),
            self.mbatch[1].to(self.device),
            # Task IDs are integers, not tensors
            self.mbatch[2],
        )

    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        self.adapted_dataset = (
            self.adapted_dataset.remove_current_transform_group()
        )

    def make_train_dataloader(
        self,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        **kwargs,
    ):
        self.dataloader = DataLoader(
            self.adapted_dataset,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=_sup_contrast_collate_fn,
        )

    def model_adaptation(self, model=None):
        _base_model = super().model_adaptation(model=model)
        _proj_head = nn.Linear(160, self.proj_head_dim).to(self.device)
        _base_model = self.model
        return nn.ModuleDict(
            {"base_model": _base_model, "proj_head": _proj_head}
        )

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
                _features = self.model.base_model.forward_features(self.mb_x)
            else:
                _task_id = self.experience.current_experience
                _progress = mb_it / (_num_mb - 1)
                _mask_scale = get_hat_mask_scale(
                    # FIXME: change this back to `cosine`
                    strat="linear",
                    progress=_progress,
                    max_trn_mask_scale=self.hat_config.max_trn_mask_scale,
                )
                _pld = HATPayload(
                    data=self.mb_x,
                    task_id=_task_id,
                    mask_scale=_mask_scale,
                )
                _features = self.model.base_model.forward_features(_pld)
            _proj = self.model.proj_head(_features)
            # Reshape from [2*bsz, proj_head_dim] to [bsz, 2, proj_head_dim]
            _proj = _proj.view(2, len(self.mb_y), -1).swapaxes(0, 1)
            self.mb_output = _proj.contiguous()
            self._after_forward(**kwargs)

            # Loss & Backward
            # `self.criterion` is a `SupContrastLoss` instance, and the
            # input are `self.mb_output` and `self.mb_y`
            self.loss = self.criterion()
            if self.hat_config is not None:
                # noinspection PyUnboundLocalVariable
                _reg_term = get_hat_reg_term(
                    module=self.model.base_model,
                    strat="uniform",
                    task_id=_task_id,
                    mask_scale=_mask_scale,
                )
                self.loss = self.loss + _reg_term
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    # def _after_training_epoch(self, **kwargs):
    #     # Report the HAT mask usage
    #     if self.hat_config is not None and self.verbose:
    #         _mask_util_df = get_hat_util(module=self.model)
    #         print(_mask_util_df)

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        # Remove the projection head of the last experience
        if isinstance(self.model, nn.ModuleDict):
            del self.model["proj_head"]
            self.model = self.model["base_model"]
        # Report the HAT mask usage
        if self.hat_config is not None and self.verbose:
            _mask_util_df = get_hat_util(module=self.model)
            print(_mask_util_df)
