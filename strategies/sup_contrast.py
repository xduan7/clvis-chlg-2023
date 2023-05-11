from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from .base import BaseStrategy

# Original SupContrast augmentation
_aug_tsfm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
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

    @torch.compile
    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].

        Returns:
            A loss scalar.

        """
        device = features.device
        batch_size = features.shape[0]
        # elif labels is not None and mask is None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float()

        # Features must be normalized similar to `NTXentLoss`
        features = nn.functional.normalize(features, dim=-1, eps=self.eps)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # elif self.contrast_mode == "all":
        anchor_feature = contrast_feature
        anchor_count = contrast_count

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
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
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


class SupContrast(BaseStrategy):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int,
        train_epochs: int,
        hat_reg_base_factor: float,
        hat_reg_decay_exp: float,
        hat_reg_enrich_ratio: float,
        num_replay_samples_per_batch: int,
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
            freeze_hat=False,
            hat_reg_base_factor=hat_reg_base_factor,
            hat_reg_decay_exp=hat_reg_decay_exp,
            hat_reg_enrich_ratio=hat_reg_enrich_ratio,
            num_replay_samples_per_batch=num_replay_samples_per_batch,
            device=device,
            plugins=plugins,
            verbose=verbose,
        )
        self.proj_head_dim = proj_head_dim

    # TODO: rotated head

    @staticmethod
    def _get_criterion():
        return SupContrastLoss()

    @staticmethod
    def _get_evaluator(verbose: bool):
        return EvaluationPlugin(
            loss_metrics(
                minibatch=False, epoch=True, experience=True, stream=True
            ),
            loggers=[InteractiveLogger()] if verbose else None,
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
            pin_memory=pin_memory if self.device.type == "cuda" else False,
            pin_memory_device=str(self.device),
            persistent_workers=persistent_workers,
            collate_fn=_sup_contrast_collate_fn,
            # drop_last=True,
        )

    def _construct_replay_tensors(self):
        # Contrastive learning requires a different set of features
        # like [sample_1_features, sample_2_features] for each sample
        # in the minibatch. We need to construct these tensors for
        # replay.
        if not self.replay or len(self.replay_features) == 0:
            return

        __logits, __targets = [], []
        for __c, __l in self.replay_features.items():
            assert len(__l) == 2, (
                "The implementation is based on 2 samples "
                "for each class for now."
            )
            __logits.append(__l)
            __targets.append(__c)
        self.replay_feature_tensor = torch.stack(__logits)
        self.replay_target_tensor = torch.tensor(__targets, device=self.device)

    def model_adaptation(self, model=None):
        self._del_replay_features(self.experience.classes_in_this_experience)
        self._construct_replay_tensors()

        _base_model = super().model_adaptation(model=model)
        _proj_head = nn.Linear(160, self.proj_head_dim).to(self.device)
        _base_model = self.model

        return nn.ModuleDict(
            {"base_model": _base_model, "proj_head": _proj_head}
        )

    def training_epoch(self, **kwargs):
        _num_mb = len(self.dataloader)
        for mb_it, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad(set_to_none=True)
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            _features = self.forward_(
                model=self.model.base_model,
                images=self.mb_x,
                mb_it=mb_it,
                return_features=True,
            )
            __replay_features, __replay_targets = self._get_replay_samples()
            if __replay_features is not None:
                __replay_features = __replay_features.reshape(-1, 160)
                _features = torch.cat([_features, __replay_features], dim=0)
                self.mbatch = (
                    self.mbatch[0],
                    torch.cat([self.mb_y, __replay_targets], dim=0),
                    self.mbatch[2],
                )
            _proj = self.model.proj_head(_features)
            # Reshape from [2*bsz, proj_head_dim] to [bsz, 2, proj_head_dim]
            _proj = _proj.view(2, len(self.mb_y), -1).swapaxes(0, 1)
            self.mb_output = _proj.contiguous()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss = self.criterion()
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
