# This file implements the slimmed ResNet with hard attention to the task,"""
from copy import deepcopy

import torch
import torch.nn as nn
from math import ceil
from torch.nn.functional import avg_pool2d

from hat import HATPayload, HATConfig
from hat.modules import HATConv2d, TaskIndexedBatchNorm2d

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin


def conv3x3(in_planes, out_planes, hat_config, stride=1):
    return HATConv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        hat_config=hat_config,
        stride=stride,
        padding=1,
        bias=False,
    )


class HATBasicBlock(HATPayloadCarrierMixin):
    expansion = 1

    def __init__(self, in_planes, planes, hat_config, stride=1):
        super(HATBasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_planes=in_planes,
            out_planes=planes,
            hat_config=hat_config,
            stride=stride,
        )
        self.bn1 = TaskIndexedBatchNorm2d(
            num_features=planes,
            num_tasks=hat_config.num_tasks,
        )
        self.conv2 = conv3x3(
            in_planes=planes,
            out_planes=planes,
            hat_config=hat_config,
        )
        self.bn2 = TaskIndexedBatchNorm2d(
            num_features=planes,
            num_tasks=hat_config.num_tasks,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                HATConv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=1,
                    hat_config=hat_config,
                    stride=stride,
                    bias=False,
                ),
                TaskIndexedBatchNorm2d(
                    num_features=self.expansion * planes,
                    num_tasks=hat_config.num_tasks,
                ),
            )
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def zero_init_last(self):
        for __bn in self.bn2:
            nn.init.zeros_(__bn.weight)

    def forward(self, pld: HATPayload) -> HATPayload:
        shortcut = pld.forward_by(self.shortcut)

        pld = pld.forward_by(self.conv1)
        pld = pld.forward_by(self.bn1)
        pld = pld.forward_by(self.act1)

        pld = pld.forward_by(self.conv2)
        pld = pld.forward_by(self.bn2)

        pld = HATPayload(
            data=pld.unmasked_data + shortcut.data,
            masker=pld.masker,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            locked_task_ids=pld.locked_task_ids,
            prev_maskers=pld.prev_maskers,
        )
        pld = pld.forward_by(self.act2)
        return pld


class HATResNet(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            num_classes,
            nf,
            hat_config,
            num_fragments=1,
            num_ensembles=1,
    ):
        super(HATResNet, self).__init__()
        self.in_planes = nf
        self.hat_config = hat_config
        self.num_fragments = num_fragments
        self.num_ensembles = num_ensembles
        self.num_tasks_per_fragment = ceil(hat_config.num_tasks / num_fragments)
        # assert self.num_tasks_per_fragment * num_fragments == hat_config.num_tasks

        # Split the num_tasks by num_fragments
        # __c, __r = divmod(hat_config.num_tasks, num_fragments)
        self.num_tasks = [self.num_tasks_per_fragment] * num_fragments
        self.hat_configs = []
        for __num_tasks in self.num_tasks:
            self.hat_configs.append(
                HATConfig(
                    num_tasks=__num_tasks,
                    max_trn_mask_scale=hat_config.max_trn_mask_scale,
                    init_strat=hat_config.init_strat,
                    grad_comp_factor=hat_config.grad_comp_factor,
                )
            )
        # Add the ensemble modules
        self.num_tasks = self.num_tasks * num_ensembles
        self.hat_configs = self.hat_configs * num_ensembles

        self.conv1 = nn.ModuleList([
            conv3x3(
                in_planes=3,
                out_planes=nf * 1,
                hat_config=__hat_config,
            ) for __hat_config in self.hat_configs
        ])
        self.bn1 = nn.ModuleList([
            TaskIndexedBatchNorm2d(
                num_features=nf * 1,
                num_tasks=__hat_config.num_tasks,
            ) for __hat_config in self.hat_configs
        ])
        self.act1 = nn.ReLU()
        self.layer1 = nn.ModuleList([
            self._make_layer(
                block=block,
                in_planes=nf * 1,
                planes=nf * 1,
                num_blocks=num_blocks[0],
                stride=1,
                hat_config=__hat_config,
            ) for __hat_config in self.hat_configs
        ])
        self.layer2 = nn.ModuleList([
            self._make_layer(
                block=block,
                in_planes=nf * 1,
                planes=nf * 2,
                num_blocks=num_blocks[1],
                stride=2,
                hat_config=__hat_config,
            ) for __hat_config in self.hat_configs
        ])
        self.layer3 = nn.ModuleList([
            self._make_layer(
                block=block,
                in_planes=nf * 2,
                planes=nf * 4,
                num_blocks=num_blocks[2],
                stride=2,
                hat_config=__hat_config,
            ) for __hat_config in self.hat_configs
        ])
        self.layer4 = nn.ModuleList([
            self._make_layer(
                block=block,
                in_planes=nf * 4,
                planes=nf * 8,
                num_blocks=num_blocks[3],
                stride=2,
                hat_config=__hat_config,
            ) for __hat_config in self.hat_configs
        ])
        # Linear layer is shared across tasks
        self.linear = nn.Linear(
            in_features=nf * 8 * block.expansion,
            out_features=num_classes,
        )

    def _make_layer(self, block, in_planes, planes, num_blocks, stride, hat_config):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=in_planes,
                    planes=planes,
                    stride=stride,
                    hat_config=hat_config,
                ),
            )
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, pld: HATPayload) -> torch.Tensor:
        features = []
        for __e in range(self.num_ensembles):
            _pld = deepcopy(pld)
            _task_id = _pld.task_id  # 14
            _module_index = _task_id // self.num_tasks_per_fragment + __e * self.num_fragments
            _new_task_id = _task_id % self.num_tasks_per_fragment  # 2
            _pld.task_id = _new_task_id
            _pld = self.conv1[_module_index](_pld)
            _pld = self.bn1[_module_index](_pld)
            _pld = _pld.forward_by(self.act1)
            _pld = self.layer1[_module_index](_pld)
            _pld = self.layer2[_module_index](_pld)
            _pld = self.layer3[_module_index](_pld)
            _pld = self.layer4[_module_index](_pld)
            _features = avg_pool2d(_pld.data, 4)
            _features = _features.view(_features.size(0), -1)
            features.append(_features)
        # Take the average of the features
        features = torch.stack(features, dim=0).mean(dim=0)
        return features

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    def forward(self, pld: HATPayload) -> torch.Tensor:
        features = self.forward_features(pld)
        logits = self.forward_head(features)
        return logits

    def get_hat_reg_term(self, task_id: int, mask_scale: float) -> torch.Tensor:
        from hat.modules import HATMasker

        _reg, _cnt = 0.0, 0
        for __e in range(self.num_ensembles):
            _module_index = task_id // self.num_tasks_per_fragment + __e * self.num_fragments
            _new_task_id = task_id % self.num_tasks_per_fragment
            for module in [
                self.conv1[_module_index],
                self.layer1[_module_index],
                self.layer2[_module_index],
                self.layer3[_module_index],
                self.layer4[_module_index],
            ]:
                for __m in module.modules():
                    if isinstance(__m, HATMasker):
                        _reg += __m.get_reg_term(
                            strat="uniform",
                            task_id=_new_task_id,
                            mask_scale=mask_scale,
                        )
                        _cnt += 1
        return _reg / _cnt if _cnt > 0 else 0.0

    @staticmethod
    def _copy_weights(src_module: nn.Module, dst_module: nn.Module):
        # Copy all the weights except for HATMasker
        from hat.modules import HATMasker

        src_dict = src_module.state_dict()
        dst_dict = dst_module.state_dict()
        # Find keys to exclude
        keys_to_exclude = []
        for __k, __m in src_module.named_modules():
            if isinstance(__m, HATMasker):
                for __pn in __m.state_dict():
                    keys_to_exclude.append(f"{__k}.{__pn}")

        for __k in src_dict:
            if __k not in keys_to_exclude and __k in dst_dict:
                dst_dict[__k].copy_(src_dict[__k])

        dst_module.load_state_dict(dst_dict)

    def copy_weights_from_previous_fragment(self, task_id: int):
        _new_task_id = task_id % self.num_tasks_per_fragment

        if _new_task_id != 0:
            return

        for __e in range(self.num_ensembles):
            _module_index = task_id // self.num_tasks_per_fragment + __e * self.num_fragments
            _prev_module_index = _module_index - 1

            if _prev_module_index < 0:
                continue

            # print(f"Copying weights from {_prev_module_index} to "
            #       f"{_module_index} for ensemble {__e}.")

            for __m in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:

                # print(f"Copying weights of {__m} ...")

                self._copy_weights(
                    src_module=__m[_prev_module_index],
                    dst_module=__m[_module_index],
                )


def HATSlimResNet18(n_classes, hat_config, nf=20, num_fragments=1,
                    num_ensembles=1):
    return HATResNet(HATBasicBlock, [2, 2, 2, 2], num_classes=n_classes,
                     nf=nf, hat_config=hat_config,
                     num_fragments=num_fragments, num_ensembles=num_ensembles)


__all__ = ["HATResNet", "HATSlimResNet18"]
