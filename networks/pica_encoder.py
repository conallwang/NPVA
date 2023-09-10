from turtle import forward
import torch
import torch.nn as nn

from networks.layers import ConvDownBlock


class TexHead(nn.Module):
    def __init__(self, res=False):
        super(TexHead, self).__init__()

        self.downsample = nn.Sequential(ConvDownBlock(3, 512, res), ConvDownBlock(512, 256, res))

    def forward(self, x):
        out = self.downsample(x)
        return out


class GeomHead(nn.Module):
    def __init__(self):
        super(GeomHead, self).__init__()

        self.conv = nn.Conv2d(3, 256, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class PiCAEncoder(nn.Module):
    def __init__(self, res=False):
        super(PiCAEncoder, self).__init__()

        self.texHead = TexHead(res)
        self.geomHead = GeomHead()

        self.downsample = nn.Sequential(
            ConvDownBlock(512, 128, res),
            ConvDownBlock(128, 64, res),
            ConvDownBlock(64, 32, res),
            ConvDownBlock(32, 16, res),
            ConvDownBlock(16, 8, res),
        )

        self.meanBatch = nn.Conv2d(8, 4, 1)
        self.varBatch = nn.Conv2d(8, 4, 1)

    def forward(self, avgTex, posMap):
        texFeat = self.texHead(avgTex)
        geomFeat = self.geomHead(posMap)

        assert texFeat.shape == geomFeat.shape  # BX256x256x256 (BxCxHxW)

        out = self.downsample(torch.concat([texFeat, geomFeat], dim=1))

        mean = self.meanBatch(out)
        variance = self.varBatch(out)

        return mean, variance
