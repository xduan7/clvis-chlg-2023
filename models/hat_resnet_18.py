# This file implements the slimmed ResNet with hard attention to the task,"""
import torch
import torch.nn as nn
from torch.nn.functional import avg_pool2d

from hat import HATPayload
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
    def __init__(self, block, num_blocks, num_classes, nf, hat_config):
        super(HATResNet, self).__init__()
        self.in_planes = nf
        self.hat_config = hat_config

        self.conv1 = conv3x3(
            in_planes=3,
            out_planes=nf * 1,
            hat_config=hat_config,
        )
        self.bn1 = TaskIndexedBatchNorm2d(
            num_features=nf * 1,
            num_tasks=hat_config.num_tasks,
        )
        self.act1 = nn.ReLU()
        self.layer1 = self._make_layer(
            block=block,
            planes=nf * 1,
            num_blocks=num_blocks[0],
            stride=1,
            hat_config=hat_config,
        )
        self.layer2 = self._make_layer(
            block=block,
            planes=nf * 2,
            num_blocks=num_blocks[1],
            stride=2,
            hat_config=hat_config,
        )
        self.layer3 = self._make_layer(
            block=block,
            planes=nf * 4,
            num_blocks=num_blocks[2],
            stride=2,
            hat_config=hat_config,
        )
        self.layer4 = self._make_layer(
            block=block,
            planes=nf * 8,
            num_blocks=num_blocks[3],
            stride=2,
            hat_config=hat_config,
        )
        # Linear layer is shared across tasks
        self.linear = nn.Linear(
            in_features=nf * 8 * block.expansion,
            out_features=num_classes,
        )

    def _make_layer(self, block, planes, num_blocks, stride, hat_config):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    hat_config=hat_config,
                ),
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, pld: HATPayload) -> torch.Tensor:
        pld = self.conv1(pld)
        pld = self.bn1(pld)
        pld = pld.forward_by(self.act1)
        pld = self.layer1(pld)
        pld = self.layer2(pld)
        pld = self.layer3(pld)
        pld = self.layer4(pld)
        features = avg_pool2d(pld.data, 4)
        features = features.view(features.size(0), -1)
        return features

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    def forward(self, pld: HATPayload) -> torch.Tensor:
        features = self.forward_features(pld)
        logits = self.forward_head(features)
        return logits


def HATSlimResNet18(n_classes, hat_config, nf=20):
    return HATResNet(HATBasicBlock, [2, 2, 2, 2], n_classes, nf, hat_config)


__all__ = ["HATResNet", "HATSlimResNet18"]
