# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nvdiffrast batched rendering
import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import os, sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from dataset import Dataset
from PIL import Image
from utils import batched_index_select, upTo8Multiples, visDepthMap, visPositionMap
from dataset.pica_dataset import load_krt, load_obj

# DEBUG
# import Metashape


class Renderer:
    def __init__(self, img_h, img_w):
        # self.glctx = dr.RasterizeGLContext()
        self.glctx = dr.RasterizeCudaContext()

        self.resolution = [img_h, img_w]

    def render(self, M, pos, pos_idx, attr=None):
        ones = torch.ones((pos.shape[0], pos.shape[1], 1)).to(pos.device)
        pos_homo = torch.cat((pos, ones), -1)
        projected = torch.bmm(M, pos_homo.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)  # [B, N_verts, 3]
        proj = torch.zeros_like(projected)
        proj[..., 0] = (projected[..., 0] / (self.resolution[1] / 2) - projected[..., 2]) / projected[..., 2]
        proj[..., 1] = (projected[..., 1] / (self.resolution[0] / 2) - projected[..., 2]) / projected[..., 2]
        clip_space, _ = torch.max(projected[..., 2], 1, keepdim=True)
        proj[..., 2] = projected[..., 2] / clip_space

        pos_view = torch.cat((proj, torch.ones(proj.shape[0], proj.shape[1], 1).to(proj.device)), -1)
        pos_idx_flat = pos_idx.view(
            (-1, 3)
        ).contiguous()  # 至于这里为什么是reshape为[-1, 3],可以去nvdiffrast的document里面搜索instanced mode

        rast_out, _ = dr.rasterize(self.glctx, pos_view, pos_idx_flat, self.resolution)

        z = projected[..., 2:]
        rast_in = z
        if attr is not None:
            rast_in = torch.cat([z, attr], dim=-1)
        rast_attr, _ = dr.interpolate(rast_in, rast_out, pos_idx_flat)

        return rast_out, rast_attr
