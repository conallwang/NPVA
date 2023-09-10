import torch
import torch.nn
import torch.nn.functional as F
import os
import numpy as np
from numpy import dot
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import time
from .render_utils import (
    mix_ray_generation,
    near_far_linear_ray_generation,
    near_far_disparity_linear_ray_generation,
    around_surface_ray_generation,
    dcenter_ray_generation,
)

parent_dir = os.path.dirname(os.path.abspath(__file__))


from torch.utils.cpp_extension import load as load_cuda

query_worldcoords_cuda = load_cuda(
    name="query_worldcoords_cuda",
    sources=[os.path.join(parent_dir, path) for path in ["cuda/query_worldcoords.cpp", "cuda/query_worldcoords.cu"]],
    verbose=True,
)


class lighting_fast_querier:
    def __init__(self, config, is_val):
        self.device = "cuda"
        self.config = config
        self.is_val = is_val
        # self.inverse = self.opt.inverse
        # self.count=0
        self.radius_limit_np = np.asarray(
            config["rendering.radius_limit_scale"] * max(config["rendering.vsize"][0], config["rendering.vsize"][1])
        ).astype(np.float32)
        self.vscale_np = np.array(config["rendering.vscale"], dtype=np.int32)
        self.scaled_vsize_np = (config["rendering.vsize"] * self.vscale_np).astype(np.float32)
        self.scaled_vsize_tensor = torch.as_tensor(self.scaled_vsize_np).cuda()
        self.kernel_size = np.asarray(config["rendering.kernel_size"], dtype=np.int32)
        self.kernel_size_tensor = torch.as_tensor(self.kernel_size).cuda()
        self.query_size = np.asarray(config["rendering.query_size"], dtype=np.int32)
        self.query_size_tensor = torch.as_tensor(self.query_size).cuda()

    def clean_up(self):
        pass

    def get_hyperparameters(self, vsize_np, point_xyz_w_tensor, ranges=None):
        """
        :param l:
        :param h:
        :param w:
        :param zdim:
        :param ydim:
        :param xdim:
        :return:
        """
        min_xyz, max_xyz = torch.min(point_xyz_w_tensor, dim=-2)[0][0], torch.max(point_xyz_w_tensor, dim=-2)[0][0]
        if ranges is not None:
            ranges_min = torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)
            ranges_max = torch.as_tensor(ranges[3:], dtype=torch.float32, device=min_xyz.device)
            min_xyz, max_xyz = (
                torch.max(torch.stack([min_xyz, ranges_min], dim=0), dim=0)[0],
                torch.min(torch.stack([max_xyz, ranges_max], dim=0), dim=0)[0],
            )
        min_xyz = min_xyz - torch.as_tensor(
            self.scaled_vsize_np * self.config["rendering.kernel_size"] / 2, device=min_xyz.device, dtype=torch.float32
        )
        max_xyz = max_xyz + torch.as_tensor(
            self.scaled_vsize_np * self.config["rendering.kernel_size"] / 2, device=min_xyz.device, dtype=torch.float32
        )

        ranges_tensor = torch.cat([min_xyz, max_xyz], dim=-1)
        vdim_np = (max_xyz - min_xyz).detach().cpu().numpy() / vsize_np
        scaled_vdim_np = np.ceil(vdim_np / self.vscale_np).astype(np.int32)
        return ranges_tensor, vsize_np, scaled_vdim_np

    def query_points(
        self,
        pixel_idx_tensor,
        point_xyz_w_tensor,
        actual_numpoints_tensor,
        near_depth,
        far_depth,
        ray_dirs_tensor,
        cam_pos_tensor,
        cam_rot_tensor,
        sampling_center,
    ):
        near_depth, far_depth = np.asarray(near_depth).item(), np.asarray(far_depth).item()
        ranges_tensor, vsize_np, scaled_vdim_np = self.get_hyperparameters(
            self.config["rendering.vsize"], point_xyz_w_tensor, ranges=self.config.get("rendering.ranges", None)
        )

        if self.config["rendering.sampling_method"] == "disp_uniform":
            raypos_tensor = near_far_disparity_linear_ray_generation(
                cam_pos_tensor,
                ray_dirs_tensor,
                self.config["rendering.z_depth_dim"],
                near=near_depth,
                far=far_depth,
                jitter=0 if self.is_val else 0.3,
            )
        elif self.config["rendering.sampling_method"] == "depth_uniform":
            raypos_tensor = near_far_linear_ray_generation(
                cam_pos_tensor,
                ray_dirs_tensor,
                self.config["rendering.z_depth_dim"],
                near=near_depth,
                far=far_depth,
                jitter=0 if self.is_val else 0.3,
            )
        elif self.config["rendering.sampling_method"] == "around_surface":
            raypos_tensor = around_surface_ray_generation(
                cam_pos_tensor,
                ray_dirs_tensor,
                self.config["rendering.z_depth_dim"],
                sampling_center,
                self.config["rendering.sampling_radius"],
                jitter=0 if self.is_val else 0.3,
            )
        elif self.config["rendering.sampling_method"] == "dcenter":
            raypos_tensor = dcenter_ray_generation(
                cam_pos_tensor,
                ray_dirs_tensor,
                self.config["rendering.z_depth_dim"],
                sampling_center,
                self.config["rendering.sampling_radius"],
                self.config["rendering.dthres"],
                jitter=0 if self.is_val else 0.3,
            )
        elif self.config["rendering.sampling_method"] == "mix":
            raypos_tensor = mix_ray_generation(
                cam_pos_tensor,
                ray_dirs_tensor,
                self.config["rendering.z_depth_dim"],
                self.config["rendering.linear_ratio"],
                near_depth,
                far_depth,
                sampling_center,
                self.config["rendering.sampling_radius"],
                jitter=0 if self.is_val else 0.3,
            )

        D = raypos_tensor.shape[2]
        R = pixel_idx_tensor.reshape(point_xyz_w_tensor.shape[0], -1, 2).shape[1]

        (
            sample_pidx_tensor,
            sample_loc_w_tensor,
            ray_mask_tensor,
            occ_numpoints,
            occ_2_pnts,
            coor_occ,
        ) = query_worldcoords_cuda.woord_query_grid_point_index(
            pixel_idx_tensor,
            raypos_tensor,
            point_xyz_w_tensor,
            actual_numpoints_tensor,
            self.kernel_size_tensor,
            self.query_size_tensor,
            self.config["rendering.SR"],
            self.config["rendering.K"],
            R,
            D,
            torch.as_tensor(scaled_vdim_np, device=self.device),
            self.config["rendering.max_o"],
            self.config["rendering.P"],
            self.radius_limit_np,
            ranges_tensor,
            self.scaled_vsize_tensor,
            self.config["rendering.gpu_maxthr"],
            self.config["rendering.NN"],
        )

        sample_ray_dirs_tensor = (
            torch.masked_select(ray_dirs_tensor, ray_mask_tensor[..., None] > 0)
            .reshape(ray_dirs_tensor.shape[0], -1, 3)[..., None, :]
            .expand(-1, -1, self.config["rendering.SR"], -1)
            .contiguous()
        )

        sample_loc = self.w2pers(sample_loc_w_tensor, cam_rot_tensor, cam_pos_tensor)

        return (
            sample_pidx_tensor,
            sample_loc,
            sample_loc_w_tensor,
            sample_ray_dirs_tensor,
            ray_mask_tensor,
            vsize_np,
            ranges_tensor.detach().cpu().numpy(),
        )

    def w2pers(self, point_xyz_w, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3
        xyz_w_shift = point_xyz_w - campos[:, None, :]
        xyz_c = torch.sum(xyz_w_shift[..., None, :] * torch.transpose(camrotc2w, 1, 2)[:, None, None, ...], dim=-1)
        z_pers = xyz_c[..., 2]
        x_pers = xyz_c[..., 0] / (xyz_c[..., 2] + 1e-9)
        y_pers = xyz_c[..., 1] / (xyz_c[..., 2] + 1e-9)
        return torch.stack([x_pers, y_pers, z_pers], dim=-1)
