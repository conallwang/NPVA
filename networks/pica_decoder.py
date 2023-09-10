import torch
import torch.nn as nn

from networks.layers import ConvUpBlock


class GeomDecoder(nn.Module):
    def __init__(self):
        super(GeomDecoder, self).__init__()

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, 32, base),
            ConvUpBlock(32, 16, base * 2),
            ConvUpBlock(16, 16, base * 4),
            ConvUpBlock(16, 8, base * 8),
            ConvUpBlock(8, 3, base * 16),  # add a block
        )

    def forward(self, x):
        return self.upsample(x)


class GeomDisDecoder1k(nn.Module):
    def __init__(self):
        super(GeomDisDecoder1k, self).__init__()

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, 64, base),
            ConvUpBlock(64, 32, base * 2),
            ConvUpBlock(32, 32, base * 4),
            ConvUpBlock(32, 16, base * 8),
            ConvUpBlock(16, 16, base * 16),  # add a block
            ConvUpBlock(16, 8, base * 32),
            ConvUpBlock(8, 3, base * 64),
        )

    def forward(self, x):
        return self.upsample(x)


class GeomDisDecoder256(nn.Module):
    def __init__(self):
        super(GeomDisDecoder256, self).__init__()

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, 32, base),
            ConvUpBlock(32, 16, base * 2),
            ConvUpBlock(16, 16, base * 4),
            ConvUpBlock(16, 8, base * 8),
            ConvUpBlock(8, 3, base * 16),  # add a block
        )

    def forward(self, x):
        return self.upsample(x)


class ExprDecoder1k(nn.Module):
    def __init__(self, num_feats):
        super(ExprDecoder1k, self).__init__()

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, num_feats * 16, base),
            ConvUpBlock(num_feats * 16, num_feats * 8, base * 2),
            ConvUpBlock(num_feats * 8, num_feats * 8, base * 4),
            ConvUpBlock(num_feats * 8, num_feats * 4, base * 8),
            ConvUpBlock(num_feats * 4, num_feats * 4, base * 16),  # add a block
            ConvUpBlock(num_feats * 4, num_feats * 2, base * 32),
            ConvUpBlock(num_feats * 2, num_feats, base * 64),
        )

    def forward(self, x):
        return self.upsample(x)


class ExprDecoder256(nn.Module):
    def __init__(self, num_feats):
        super(ExprDecoder256, self).__init__()

        base = 8
        self.upsample = nn.Sequential(
            ConvUpBlock(4, num_feats * 8, base),
            ConvUpBlock(num_feats * 8, num_feats * 4, base * 2),
            ConvUpBlock(num_feats * 4, num_feats * 4, base * 4),
            ConvUpBlock(num_feats * 4, num_feats * 2, base * 8),
            ConvUpBlock(num_feats * 2, num_feats, base * 16),  # add a block
        )

    def forward(self, x):
        return self.upsample(x)


class PiCADecoder(nn.Module):
    def __init__(self, num_feats, use_disp=True, upsample=True):
        super(PiCADecoder, self).__init__()

        self.use_disp = use_disp
        ExprDecoder = ExprDecoder1k if upsample else ExprDecoder256
        GeomDisDecoder = GeomDisDecoder1k if upsample else GeomDisDecoder256

        self.geomDecoder = GeomDecoder()
        self.exprDecoder = ExprDecoder(num_feats)

        if use_disp:
            self.geomDisDecoder = GeomDisDecoder()

    def forward(self, z_code):
        """_summary_

        Args:
            z_code (Bx4x8x8): latent code sampled from gaussion
            view (Bx3x8x8): view vector
        """
        # exprFeats = torch.cat([z_code, view], dim=1)
        posMap = self.geomDecoder(z_code)
        exprCode = self.exprDecoder(z_code)

        if self.use_disp:
            dispMap = self.geomDisDecoder(z_code)
        else:
            dispMap = None

        return posMap, dispMap, exprCode
