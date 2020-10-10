import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.block import (
    GenericRSUBlock,
    GenericRSUFBlock, _upsample_like
)

__all__ = ['U2SquaredNet', 'BigU2Net']


class U2SquaredNet(nn.Module):
    r"""
    Built U^2 Net using generic blocks. Structure as defined here:
    https://github.com/NathanUA/U-2-Net/blob/ca2562585b5dfd51cb4b54e4726c7addbbaf2af2/model/u2net.py#L424
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U2SquaredNet, self).__init__()

        # encoder
        self.stage1 = GenericRSUBlock(in_ch, 16, 64, L=7)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = GenericRSUBlock(64, 16, 64, L=6)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = GenericRSUBlock(64, 16, 64, L=5)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = GenericRSUBlock(64, 16, 64, L=4)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = GenericRSUFBlock(64, 16, 64, L=4)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = GenericRSUFBlock(64, 16, 64, L=4)

        # decoder
        self.stage5d = GenericRSUFBlock(128, 16, 64, L=4)
        self.stage4d = GenericRSUBlock(128, 16, 64, L=4)
        self.stage3d = GenericRSUBlock(128, 16, 64, L=5)
        self.stage2d = GenericRSUBlock(128, 16, 64, L=6)
        self.stage1d = GenericRSUBlock(128, 16, 64, L=7)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.out_conv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        hx1 = self.stage1.forward(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2.forward(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3.forward(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4.forward(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5.forward(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6.forward(hx)
        hx6_up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d.forward(torch.cat((hx6_up, hx5), 1))
        hx5d_up = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d.forward(torch.cat((hx5d_up, hx4), 1))
        hx4d_up = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d.forward(torch.cat((hx4d_up, hx3), 1))
        hx3d_up = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d.forward(torch.cat((hx3d_up, hx2), 1))
        hx2d_up = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d.forward(torch.cat((hx2d_up, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.out_conv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


class BigU2Net(nn.Module):
    """
    Built U^2 Net P (small version) using generic blocks. Structure as defined here:
    https://github.com/NathanUA/U-2-Net/blob/ca2562585b5dfd51cb4b54e4726c7addbbaf2af2/model/u2net.py#L319

    """
    def __init__(self, in_ch=3, out_ch=1):
        super(BigU2Net, self).__init__()

        self.stage1 = GenericRSUBlock(in_ch, 32, 64, L=7)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = GenericRSUBlock(64, 32, 128, L=6)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = GenericRSUBlock(128, 64, 256, L=5)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = GenericRSUBlock(256, 128, 512, L=4)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = GenericRSUFBlock(512, 256, 512, L=4)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = GenericRSUFBlock(512, 256, 512, L=4)

        # dec
        self.stage5d = GenericRSUFBlock(1024, 256, 512, L=4)
        self.stage4d = GenericRSUBlock(1024, 128, 256, L=4)
        self.stage3d = GenericRSUFBlock(512, 64, 128, L=5)
        self.stage2d = GenericRSUFBlock(256, 32, 64, L=6)
        self.stage1d = GenericRSUFBlock(128, 16, 64, L=7)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
