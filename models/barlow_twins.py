from typing import Union

import torch
import torch.nn as nn

from .utils import MLP

from lightly.models.modules.heads import ProjectionHead


class BarlowTwins(nn.Module):
    """
    Code from https://github.com/facebookresearch/barlowtwins
    """

    def __init__(
        self,
        backbone_net,
        projector_hidden: Union[int, tuple] = (8192, 8192, 8192),
        λ: float = 0.0051,
    ):
        super().__init__()

        self.lambd = λ

        self.backbone_net = backbone_net
        self.repre_dim = self.backbone_net.fc.in_features
        backbone_net.fc = nn.Flatten()

        self.projector = ProjectionHead(
            [
                (512, 2048, nn.BatchNorm1d(2048), nn.ReLU(inplace=True)),
                (2048, 2048, None, None),
            ]
        )

    def loss_fn(self, z1, z2, λ):
        # taken from https://github.com/lightly-ai/lightly/blob/0be5e3b7b4e0b748fe9f2c2e31d2cf6ed3c5f77c/lightly/loss/barlow_twins_loss.py#L42
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD

        n, d = z1.shape
        c = torch.mm(z1_norm.T, z2_norm) / n

        c_diff = (c - torch.eye(d, device=z1.device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(d, dtype=bool)] *= λ

        return c_diff.sum()

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone_net(x1))
        z2 = self.projector(self.backbone_net(x2))

        return self.loss_fn(z1, z2, self.lambd)
