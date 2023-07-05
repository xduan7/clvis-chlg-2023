from abc import abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from hat import HATPayload
from hat.utils import get_hat_mask_scale, get_hat_reg_term, get_hat_util


class DatasetWithTsfm(Dataset):

    def __init__(
        self,
        dataset,
        img_tsfm=None,
        trgt_tsfm=None,
    ):
        self.dataset = [list(__i) for __i in dataset]
        self.img_tsfm = img_tsfm
        self.trgt_tsfm = trgt_tsfm

    def remove_current_transform_group(self):
        self.img_tsfm = None
        self.trgt_tsfm = None
        return self

    def replace_current_transform_group(
        self, transform_group: tuple
    ):
        assert len(transform_group) == 2
        self.img_tsfm = transform_group[0]
        self.trgt_tsfm = transform_group[1]
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, trgt, exp = self.dataset[item]
        if self.img_tsfm is not None:
            img = self.img_tsfm(img)
        if self.trgt_tsfm is not None:
            trgt = self.trgt_tsfm(trgt)
        return img, trgt, exp

    def subset(self, indices):
        return DatasetWithTsfm(
            [self.dataset[i] for i in indices],
            self.img_tsfm,
            self.trgt_tsfm,
        )


class BaseStrategy(SupervisedTemplate):
    """Base strategy class for the competition.

    Contains the basic functionality for training and evaluation.

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int,
        train_epochs: int,
        freeze_hat: bool,
        hat_reg_base_factor: float,
        hat_reg_decay_exp: float,
        hat_reg_enrich_ratio: float,
        num_replay_samples_per_batch: int,
        device: torch.device,
        plugins: Optional[List[SupervisedPlugin]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            criterion=self._get_criterion(),
            device=device,
            plugins=plugins,
            evaluator=self._get_evaluator(verbose),
        )

        self.hat_config = getattr(self.model, "hat_config", None)
        self.verbose = verbose
        self.task_id = None
        self.num_mb = None
        self.mask_scale = None
        self.freeze_hat = freeze_hat
        self.hat_reg_base_factor = hat_reg_base_factor
        self.hat_reg_decay_exp = hat_reg_decay_exp
        self.hat_reg_enrich_ratio = hat_reg_enrich_ratio
        self.hat_reg_factor = None

        self.replay = num_replay_samples_per_batch > 0
        self.replay_features = {}
        self.replay_feature_tensor = None
        self.replay_target_tensor = None
        self.replay_logit_tensor = None
        self.num_replay_samples_per_class = 2
        self.num_replay_samples_per_batch = num_replay_samples_per_batch

    @staticmethod
    @abstractmethod
    def _get_criterion():
        pass

    @staticmethod
    @abstractmethod
    def _get_evaluator(verbose: bool):
        pass

    def _unpack_minibatch(self):
        self._check_minibatch()
        self.mbatch = (
            self.mbatch[0].to(self.device, non_blocking=True),
            self.mbatch[1].to(self.device, non_blocking=True),
            self.mbatch[2],
        )

    def _del_replay_features(self, classes: List[int]):
        for __c in classes:
            if __c in self.replay_features:
                del self.replay_features[__c]

    @torch.no_grad()
    def _construct_replay_tensors(self, target: Optional[int] = None):
        # Construct replay feature and target tensors
        if not self.replay or len(self.replay_features) == 0:
            return

        __logits, __targets = [], []
        for __c, (__mean, __std) in self.replay_features.items():
            # Shape of the logits are [num_samples, num_features]
            for __i in range(__mean.shape[0]):
                __l = [
                    torch.normal(__mean[__i], __std[__i]).to(self.device)
                    for _ in range(self.train_mb_size * 2)
                ]
                __l = torch.stack(__l, dim=0)
                __logits.append(__l)
                __targets += [__c if target is None else target] * len(__l)
        self.replay_feature_tensor = torch.cat(__logits, dim=0)
        self.replay_target_tensor = torch.tensor(__targets, device=self.device)

        # Construct replay logits
        self.replay_logit_tensor = self.model.linear(self.replay_feature_tensor)

    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation()
        self.adapted_dataset = (
            self.adapted_dataset.remove_current_transform_group()
        )
        # self.adapted_dataset = DatasetWithTsfm(
        #     self.adapted_dataset,
        # )

    def model_adaptation(self, model=None):
        _model = super().model_adaptation(model)
        if self.freeze_hat:
            from hat.modules.maskers import HATMasker

            for __m in _model.modules():
                if isinstance(__m, HATMasker):
                    for __p in __m.parameters():
                        __p.requires_grad = False

        # if self.replay:
        #     self.num_replay_samples_per_batch = \
        #         100 * self.train_mb_size // len(
        #             self.experience.classes_in_this_experience) - self.train_mb_size
        #     print(f"Change the number of replay samples to "
        #           f"{self.num_replay_samples_per_batch}")

        return _model

    def make_optimizer(self):
        # Recreate an optimizer with the new parameters to
        # (1) prevent momentum/gradient carry-over
        # (2) train on the new projection head
        self.optimizer = self.optimizer.__class__(
            self.model.parameters(),
            **self.optimizer.defaults,
        )

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        self.task_id = self.experience.current_experience
        self.num_mb = len(self.dataloader)
        if self.freeze_hat:
            self.hat_reg_factor = 0.0
            print("HAT is frozen (hat_reg_factor = 0.0)")
        else:
            if hasattr(self.model, "num_fragments"):
                _num_frag = self.model.num_fragments
                _num_tasks = self.model.num_tasks
            else:
                _num_frag = self.model.base_model.num_fragments
                _num_tasks = self.model.base_model.num_tasks

            _module_index = self.task_id % _num_frag
            _new_task_id = self.task_id // _num_frag
            _num_tasks_in_this_fragment = _num_tasks[_module_index]
            if _num_tasks_in_this_fragment == 1:
                __hat_reg_decay_factor = 0.0
            else:
                __hat_reg_decay_factor = (
                    (_num_tasks_in_this_fragment - _new_task_id - 1) / (_num_tasks_in_this_fragment - 1)
                ) ** self.hat_reg_decay_exp
            # If the number of classes is small, we add more regularization
            # otherwise we subtract some regularization.
            __hat_reg_enrich_ratio = (
                1
                + (len(self.experience.classes_in_this_experience) - 25)
                * self.hat_reg_enrich_ratio
                / 100
            )
            self.hat_reg_factor = (
                self.hat_reg_base_factor
                * __hat_reg_enrich_ratio
                * __hat_reg_decay_factor
            )
            # Extremely small regularization for replayed samples during
            # representation learning is the only way ...
            # if self.task_id >= 1 and hasattr(self, "proj_head_dim") and self.num_replay_samples_per_batch > 0:
            #     self.hat_reg_factor = 0.008 * self.hat_reg_factor
            print(
                f"HAT reg factor: "
                f"hat_reg_base_factor({self.hat_reg_base_factor:.2f}) * "
                f"hat_reg_enrich_ratio({__hat_reg_enrich_ratio:.2f}) * "
                f"hat_reg_decay_factor({__hat_reg_decay_factor:.2f}) = "
                f"{self.hat_reg_factor:.2f} "
            )

    def _get_replay_samples(self):
        if (not self.replay) or (self.replay_feature_tensor is None):
            return None, None, None
        elif (
            self.num_replay_samples_per_batch
            >= self.replay_feature_tensor.shape[0]
        ):
            return (
                self.replay_feature_tensor,
                self.replay_target_tensor,
                self.replay_logit_tensor,
            )
        else:
            _indices = torch.randperm(
                self.replay_feature_tensor.shape[0],
                device=self.device,
            )[: self.num_replay_samples_per_batch]
            return (
                self.replay_feature_tensor[_indices],
                self.replay_target_tensor[_indices],
                self.replay_logit_tensor[_indices],
            )

    def sync_replay_features(self, strategy):
        if not self.replay or not strategy.replay:
            return
        self.replay_features = strategy.replay_features

    def forward_(
        self,
        model: nn.Module,
        images: torch.Tensor,
        return_features: bool,
        mb_it: Optional[int] = None,
        mask_scale: Optional[float] = None,
        task_id: Optional[int] = None,
    ):
        if self.hat_config is None:
            _features = model.forward_features(images)
        else:
            if mask_scale is None:
                _progress = mb_it / (self.num_mb - 1)
                self.mask_scale = get_hat_mask_scale(
                    strat="cosine",
                    progress=_progress,
                    max_trn_mask_scale=self.hat_config.max_trn_mask_scale,
                )
            else:
                # Save mask scale for regularization term
                self.mask_scale = mask_scale

            if task_id is None:
                task_id = self.task_id

            _pld = HATPayload(
                data=images,
                task_id=task_id,
                mask_scale=self.mask_scale,
            )
            _features = model.forward_features(_pld)

        if return_features:
            return _features
        else:
            return self.model.forward_head(_features)

    def _get_hat_reg(self):
        if hasattr(self.model, "get_hat_reg_term"):
            return self.model.get_hat_reg_term(
                task_id=self.task_id,
                mask_scale=self.mask_scale,
            )
        else:
            return self.model.base_model.get_hat_reg_term(
                task_id=self.task_id,
                mask_scale=self.mask_scale,
            )

    def criterion(self):
        if self.hat_config is None:
            return super().criterion()
        elif getattr(self, "freeze_hat", False):
            return super().criterion()
        else:
            _loss = super().criterion()
            return _loss + self.hat_reg_factor * self._get_hat_reg()

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        # Report the HAT mask usage
        if (
            self.hat_config is not None
            and self.verbose
            and not self.freeze_hat
        ):
            _mask_util_df = get_hat_util(module=self.model)
            print(_mask_util_df)
        # Unfreeze everything
        for __p in self.model.parameters():
            try:
                __p.requires_grad = True
            except RuntimeError:
                pass
