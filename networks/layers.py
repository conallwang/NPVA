from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv2d with weight normalization
class Conv2dWN(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-12
    ):
        super(Conv2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.eps = eps

    def forward(self, x):
        wnorm = max(torch.sqrt(torch.sum(self.weight**2)), self.eps)
        return F.conv2d(
            x,
            self.weight * self.g[:, None, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2dBN(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dBN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out


# refer to https://github1s.com/facebookresearch/multiface/blob/HEAD/models.py#L537-L577
class Conv2dWNUB(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-12,
    ):
        super(Conv2dWNUB, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if isinstance(feature_size, int):
            feature_size = (feature_size, feature_size)
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, feature_size[0], feature_size[1]))
        self.eps = eps

    def forward(self, x):
        wnorm = max(torch.sqrt(torch.sum(self.weight**2)), self.eps)
        return (
            F.conv2d(
                x,
                self.weight * self.g[:, None, None, None] / wnorm,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class Conv2dBNUB(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dBNUB, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if isinstance(feature_size, int):
            feature_size = (feature_size, feature_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, feature_size[0], feature_size[1]))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x) + self.bias[None, ...])
        return out


# A Conv Downsample Block
class ConvDownBlock(nn.Module):
    def __init__(self, cin, cout, res=False):
        super(ConvDownBlock, self).__init__()

        # self.conv = Conv2dWN(cin, cout, 4, 2, padding=1)
        self.conv = Conv2dBN(cin, cout, 4, 2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # refer to https://github1s.com/facebookresearch/multiface/blob/HEAD/models.py Line 348
        self.res = res
        if res:
            # self.res1 = Conv2dWN(cout, cout, 3, 1, 1)
            self.res1 = Conv2dBN(cout, cout, 3, 1, 1)

    def forward(self, x):
        out = self.relu(self.conv(x))
        if self.res:
            out = self.relu(self.res1(out) + out)

        return out


# A Conv Upsample Block
class ConvUpBlock(nn.Module):
    def __init__(self, cin, cout, feature_size):
        super(ConvUpBlock, self).__init__()

        # self.conv1 = Conv2dWNUB(cin, cout, feature_size, 3, padding=1)
        self.conv1 = Conv2dBNUB(cin, cout, feature_size, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, 4 * cout, 1)

        self.relu = nn.LeakyReLU(0.2)

        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.relu(self.conv2(self.conv1(x))))
