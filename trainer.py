import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import cv2
import numpy as np
import scipy.sparse
import lpips

from torch.nn.parallel import DistributedDataParallel as DDP

from networks.pica_decoder import PiCADecoder
from networks.pica_encoder import PiCAEncoder
from networks.render import Renderer
from rendering.neural_points import NeuralPoints
from rendering.point_aggregator import PointAggregator
from rendering.render_utils import ray_march, alpha_ray_march, render_func, blend_func, tone_map
from utils import (
    AverageMeter,
    batched_index_select,
    gradient_smooth,
    hook,
    plot_grad_flow,
    posmap_sample,
    restore_model,
    timing,
    upTo8Multiples,
    update_lambda,
    upsample,
    visPositionMap,
    visDepthMap,
    downsample,
    write_obj,
)

# DEBUG
# from tools.gpu_profile import gpu_profile


class Trainer:
    def __init__(self, config, logger, train_loader=None, is_val=False):
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)

        self.models = {}
        self.parameters_to_train = []
        # Input:
        #   posMap and avgTex
        # Output:
        #   mean and variance
        self.models["encoder"] = PiCAEncoder().cuda()
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Input:
        #   latent code and view dir
        # Output:
        #   recPosMap and exprCode
        self.models["decoder"] = PiCADecoder(
            config["rendering.point_features_dim"], config["ablation.use_disp"], config["ablation.upsample"]
        ).cuda()
        self.parameters_to_train += list(self.models["decoder"].parameters())

        # Input:
        #   M, samplePos, triangles
        # Output:
        #   rast_out
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.renderer = Renderer(upTo8Multiples(config["data.img_h"]), upTo8Multiples(config["data.img_w"]))

        self.models["aggregator"] = PointAggregator(config).cuda()
        self.parameters_to_train += list(self.models["aggregator"].parameters())

        # set optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters_to_train, config["training.learning_rate"], weight_decay=config["training.weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config["training.step"], gamma=0.1
        )

        # Restore checkpoint
        checkpoint_path = (
            os.path.join(config["local_workspace"], "checkpoint_latest.pth")
            if config["training.pretrained_checkpoint_path"] is None
            else config["training.pretrained_checkpoint_path"]
        )

        self.current_epoch = 1
        self.global_step = 0
        if os.path.exists(checkpoint_path):
            self.current_epoch, self.global_step = restore_model(
                checkpoint_path, self.models, self.optimizer, train_loader, logger
            )

        if not is_val:
            process_group = torch.distributed.new_group(range(dist.get_world_size()))
            for name in self.models.keys():
                self.models[name] = nn.SyncBatchNorm.convert_sync_batchnorm(self.models[name], process_group)
                self.models[name] = DDP(self.models[name], find_unused_parameters=True)
                self.models[name].train()
        else:
            for name in self.models.keys():
                self.models[name] = nn.DataParallel(self.models[name])

        self.train_losses = {
            "loss": AverageMeter("train_loss"),
            "loss_pho/rgb": AverageMeter("train_rgb_loss"),
            "loss_pho/lpips": AverageMeter("train_lpips_loss"),
            "loss_geo/depth": AverageMeter("train_depth_loss"),
            "loss_geo/rast_depth": AverageMeter("train_rast_depth_loss"),
            "loss_geo/mesh": AverageMeter("train_mesh_loss"),
            "loss_geo/posmap_grad": AverageMeter("train_grad_smooth_loss"),
            "loss_reg/kl_loss": AverageMeter("train_kl_loss"),
            "loss_reg/disp": AverageMeter("train_disp_reg_loss"),
        }
        self.val_losses = {
            "loss": AverageMeter("val_loss"),
            "loss_pho/rgb": AverageMeter("val_rgb_loss"),
            "loss_pho/lpips": AverageMeter("val_lpips_loss"),
            "loss_pho/mse": AverageMeter("val_mse"),
            "loss_pho/psnr": AverageMeter("val_psnr"),
            "loss_geo/depth": AverageMeter("val_depth_loss"),
            "loss_geo/rast_depth": AverageMeter("train_rast_depth_loss"),
            "loss_geo/mesh": AverageMeter("val_mesh_loss"),
            "loss_geo/posmap_grad": AverageMeter("val_grad_smooth_loss"),
            "loss_reg/kl_loss": AverageMeter("val_kl_loss"),
            "loss_reg/disp": AverageMeter("val_disp_reg_loss"),
        }
        self.lpips = lpips.LPIPS(net="vgg").cuda()

        self.config = config
        self.logger = logger
        self.vis = False
        self.sample_method = "None"

        self.tb_writer = config.get("tb_writer", None)

        self.init_data()

        # sys.settrace(gpu_profile)

    def set_train(self):
        """Convert models to training mode"""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert models to evaluation mode"""
        for m in self.models.values():
            m.eval()

    def train(self, train_loader, val_loader, show_time=False):
        self.sample_method = train_loader.dataset.random_sample
        while self.current_epoch <= self.config["training.epochs"]:
            success = self.train_epoch(train_loader, val_loader, show_time)
            if not success:
                return

            self.scheduler.step()
            if self.config["global_rank"] == 0:
                self.logger.info("Epoch finished, average losses: ")
                for v in self.train_losses.values():
                    self.logger.info("    {}".format(v))

            if self.current_epoch == self.config["training.first_phase"]:
                # change to use errMap sampling
                self.sample_method = "err"
                train_loader.dataset.random_sample = "err"
                if self.config["global_rank"] == 0:
                    self.logger.info("change grid sample method to err")

            if self.current_epoch == self.config["training.second_phase"]:
                # change to use errMap sampling
                self.sample_method = "patch"
                train_loader.dataset.random_sample = "patch"
                if self.config["global_rank"] == 0:
                    self.logger.info("change err sample method to patch")
            self.current_epoch += 1

    def init_data(self):
        B, H, W = self.config["data.per_gpu_batch_size"], self.config["data.img_h"], self.config["data.img_w"]
        tex_size = self.config["data.tex_size"]
        pos_size = self.config["data.pos_size"]

        self.avgTex = torch.zeros((B, 3, tex_size, tex_size), dtype=torch.float32).cuda()
        self.posMap = torch.zeros((B, 3, pos_size, pos_size), dtype=torch.float32).cuda()

        self.photo = torch.zeros((B, H, W, 3), dtype=torch.float32).cuda()
        self.depth_map = torch.zeros((B, H, W), dtype=torch.float32).cuda()

        self.M = torch.zeros((B, 3, 4), dtype=torch.float32).cuda()

        # neural points
        self.pixel_idx = torch.zeros((B, H, W, 2), dtype=torch.int).cuda()
        self.raydir = torch.zeros((B, H, W, 3), dtype=torch.float32).cuda()
        self.camrotc2w = torch.zeros((B, 3, 3), dtype=torch.float32).cuda()
        self.campos = torch.zeros((B, 3), dtype=torch.float32).cuda()
        self.intrinsic = torch.zeros((B, 3, 3), dtype=torch.float32).cuda()
        self.camrotf2w = torch.zeros((B, 3, 3), dtype=torch.float32).cuda()
        self.camtf2w = torch.zeros((B, 3), dtype=torch.float32).cuda()

        # fixed uv
        self.tris = torch.from_numpy(np.load(self.config["data.tris_path"])).int().cuda() - 1  # [N_face, 3]
        self.mouth_mask = torch.from_numpy(np.load(self.config["data.mouth_mask"])).bool().cuda()[:, None]  # [N_uvs, 1]

        self.posMean = (
            torch.from_numpy(
                np.rot90(np.load("{}/posMean.npy".format(self.config["data.gen_dir"])), k=1, axes=(1, 2)).copy()
            )
            .float()
            .cuda()
        )
        # self.posMean1k = (
        #     torch.from_numpy(
        #         np.rot90(np.load("{}/posMean1k.npy".format(self.config["data.gen_dir"])), k=1, axes=(1, 2)).copy()
        #     )
        #     .float()
        #     .cuda()
        # )
        self.posStd = float(np.genfromtxt("{}/posVar.txt".format(self.config["data.gen_dir"])) ** 0.5)

    def set_data(self, items, is_val=False):
        self.avgTex.resize_as_(items["avg_tex"]).copy_(items["avg_tex"])
        self.posMap.resize_as_(items["posMap"]).copy_(items["posMap"])

        if not is_val:
            self.photo.resize_as_(items["photo"]).copy_(items["photo"])
            self.depth_map.resize_as_(items["depth_map"]).copy_(items["depth_map"])
            self.idx = items["idx"].cuda()

        self.M.resize_as_(items["M"]).copy_(items["M"])

        # neural points
        self.pixel_idx.resize_as_(items["pixel_idx"]).copy_(items["pixel_idx"])
        self.raydir.resize_as_(items["raydir"]).copy_(items["raydir"])
        self.camrotc2w.resize_as_(items["camrotc2w"]).copy_(items["camrotc2w"])
        self.campos.resize_as_(items["campos"]).copy_(items["campos"])
        self.intrinsic.resize_as_(items["intrinsic"]).copy_(items["intrinsic"])
        self.camrotf2w.resize_as_(items["camrotf2w"]).copy_(items["camrotf2w"])
        self.camtf2w.resize_as_(items["camtf2w"]).copy_(items["camtf2w"])

    def gen_vert_normals(self, posMap):
        bot_2right = posMap[:, 1:, 1:] - posMap[:, 1:, :-1]  # 每个格子底部向右的向量, [B, 255, 255, 3]
        left_2bot = posMap[:, 1:, :-1] - posMap[:, :-1, :-1]
        diag_2up = posMap[:, :-1, 1:] - posMap[:, 1:, :-1]
        diag_2bot = -diag_2up

        bot_fnormals = torch.cross(diag_2bot, bot_2right)
        up_fnormals = torch.cross(left_2bot, diag_2up)

        B = posMap.shape[0]
        f_normals = torch.cat(
            (up_fnormals.reshape(B, -1, 3), bot_fnormals.reshape(B, -1, 3)), axis=1
        )  # [B, N_face, 3] 前65025个是上半边face的normals,后65025是下半边face的normals
        f_normals = F.normalize(f_normals, dim=-1)

        p2f = self.p2f[None].expand(B, -1, -1)  # [B, N_verts, 6]
        v_normals = (
            batched_index_select(f_normals, 1, p2f.reshape(B, -1)).reshape(B, -1, 6, 3).mean(dim=2)
        )  # [B, N_verts, 3]
        v_normals = F.normalize(v_normals, dim=-1)

        return v_normals

    def get_sampling_center(self, rast_depth, pixel_idx, span=16):
        px, py = pixel_idx[0, :, 0].long(), pixel_idx[0, :, 1].long()
        depth_max = rast_depth[:, py, px].clone()
        depth_min = depth_max.clone()
        depth_min[depth_min < 10] = 1e10
        for i in [-span, 0, span]:
            for j in [-span, 0, span]:
                x = torch.clip(px + i, 0, self.img_w - 1)
                y = torch.clip(py + j, 0, self.img_h - 1)
                cur_depth = rast_depth[:, y, x]
                depth_max = torch.where(depth_max >= cur_depth, depth_max, cur_depth)

                cur_depth[cur_depth < 10] = 1e10
                depth_min = torch.where(depth_min <= cur_depth, depth_min, cur_depth)
        return torch.cat([depth_max, depth_min], dim=-1)  # [B, N, 2]

    def encode(self, avgTex, posMap, is_val=False, debug=False):
        # refer to https://github1s.com/facebookresearch/multiface/blob/HEAD/models.py line 100
        # predict log std
        if debug:
            st = time.time()

        mean, logstd = self.models["encoder"](avgTex, posMap)
        mean = mean * 0.1
        logstd = logstd * 0.01
        if is_val:
            z_code = mean
        else:
            std = torch.exp(logstd)
            eps = torch.randn_like(mean)
            z_code = mean + std * eps  # sample latent code

        if debug:
            timing(self.logger, st, label="[Encoder]")

        return {"logstd": logstd, "mean": mean, "z_code": z_code}

    def net_decode(self, z_code, debug=False):
        if debug:
            st = time.time()

        posMap, dispMap, feats = self.models["decoder"](z_code)
        posMap = posMap * self.posStd + self.posMean

        B = posMap.shape[0]
        if self.config["ablation.upsample"]:
            scale = self.config["ablation.upscale"]
            posMap1k = F.interpolate(posMap, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            posMap1k = posMap  # [1, 3, 256, 256]

        if self.config["ablation.use_disp"]:
            verts = (posMap1k + dispMap).reshape(B, 3, -1).permute((0, 2, 1))  # [B, 1k*1k, 3]
        else:
            verts = posMap1k.reshape(B, 3, -1).permute((0, 2, 1))
        verts_w = verts @ self.camrotf2w.transpose(1, 2) + self.camtf2w.unsqueeze(1)
        # for debug:
        down4feats = downsample(feats).reshape(B, self.config["rendering.point_features_dim"], -1).permute(
            (0, 2, 1)
        )
        
        feats = feats.reshape(B, self.config["rendering.point_features_dim"], -1).permute(
            (0, 2, 1)
        )  # [B, 65536, N], cur: N=32

        if debug:
            timing(self.logger, st, label="[Net Decoder]")

        return {
            "posMap": posMap,
            "posMap1k": posMap1k,
            "dispMap": dispMap,
            "verts": verts,
            "verts_w": verts_w,
            "feats": feats,
            "down4feats": down4feats
        }

    def decode(self, z_code):
        dec_res = self.net_decode(z_code)

        # nvdiffrast: rasterization
        posMap = dec_res["posMap"]
        rast_res = self.diff_rasterization(posMap.reshape(posMap.shape[0], 3, -1).permute((0, 2, 1)), full=True)

        # neural points ray marching
        verts_w, feats = dec_res["verts_w"], dec_res["feats"]
        rast_depth = rast_res["rast_depth"]

        render_rgb = torch.zeros_like(self.photo).to(self.photo.device)
        render_depth = torch.zeros_like(self.depth_map).to(self.depth_map.device)
        # v1: using for loop to process batch
        neural_points = NeuralPoints(self.config, verts_w[0], feats[0])

        pixel_idx = self.pixel_idx
        raydir = self.raydir
        camrotc2w = self.camrotc2w
        campos = self.campos
        near = torch.tensor([self.config["rendering.near"]]).cuda()
        far = torch.tensor([self.config["rendering.far"]]).cuda()
        rast_depth_pred = rast_depth[:, pixel_idx[0, :, 1].long(), pixel_idx[0, :, 0].long(), 0]
        with torch.no_grad():
            if self.config["rendering.sampling_method"] == "dcenter":
                sampling_center = self.get_sampling_center(
                    rast_depth.clone(), pixel_idx, span=self.config["rendering.span"]
                )
            else:
                sampling_center = rast_depth_pred.clone()

        (
            sampled_color,
            sampled_Rw2c,
            sampled_dir,
            sampled_conf,
            sampled_embedding,
            sampled_xyz_pers,
            sampled_xyz,
            sample_pnt_mask,
            sample_loc,
            sample_loc_w,
            sample_ray_dirs,
            ray_mask_tensor,
            vsize,
            grid_vox_sz,
        ) = neural_points(
            {
                "pixel_idx": pixel_idx,
                "camrotc2w": camrotc2w,
                "campos": campos,
                "near": near,
                "far": far,
                "raydir": raydir,
                "sampling_center": sampling_center.detach(),
            }
        )

        data_invalid = torch.tensor(0).cuda()
        if sampled_embedding.shape[1] == 0:
            data_invalid = torch.tensor(1).cuda()
        dist.all_reduce(data_invalid, op=dist.ReduceOp.SUM)
        if data_invalid > 0:
            if self.config["global_rank"] == 0:
                self.logger.info("Have a invalid data, skip. (all rays hit no points)")
            return {}

        decoded_features, ray_valid, weight, conf_coefficient = self.models["aggregator"](
            sampled_color,
            sampled_Rw2c,
            sampled_dir,
            sampled_conf,
            sampled_embedding,
            sampled_xyz_pers,
            sampled_xyz,
            sample_pnt_mask,
            sample_loc,
            sample_loc_w,
            sample_ray_dirs,
            vsize,
            grid_vox_sz,
        )

        # point-nerf 计算 sample points 间距离
        # ray_dist = torch.cummax(sample_loc[..., 2], dim=-1)[0]
        # ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize[2], device=ray_dist.device)], dim=-1)

        # mask = ray_dist < 1e-8
        # if self.config["rendering.raydist_mode_unit"] > 0:
        #     mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
        # mask = mask.to(torch.float32)
        # ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
        # ray_dist *= ray_valid.float()

        # My Implementation
        z_vals = torch.cummax(sample_loc[..., 2], dim=-1)[0]  # [B, R, SR]
        ray_dist = torch.cat(
            [
                z_vals[..., 1:] - z_vals[..., :-1],
                torch.zeros((z_vals.shape[0], z_vals.shape[1], 1), device=z_vals.device),
            ],
            dim=-1,
        )

        last_valid_idx = torch.cumsum(ray_valid, dim=-1).argmax(-1)[0].long()  # [R]
        rays = torch.tensor(range(last_valid_idx.shape[0])).long().to(last_valid_idx.device)
        ray_dist[:, rays, last_valid_idx] = 1e10

        # mask = ray_dist < 1e-8
        # if self.config["rendering.raydist_mode_unit"] > 0:
        #     mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
        # mask = mask.to(torch.float32)
        # ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
        # ray_dist *= ray_valid.float()

        bg_color = None
        (ray_color, point_color, opacity, acc_transmission, blend_weight) = ray_march(
            ray_dist, ray_valid, decoded_features, render_func, blend_func, bg_color
        )
        ray_color = tone_map(ray_color)
        ray_depth = (z_vals * blend_weight[..., 0]).sum(-1)

        valid_mask = ray_mask_tensor[0] > 0
        render_rgb[0, valid_mask] = ray_color
        render_depth[0, valid_mask] = ray_depth

        return {
            "posMap": posMap,
            "dispMap": dec_res["dispMap"],
            "verts": dec_res["verts"],
            "render_rgb": render_rgb,
            "render_depth": render_depth,
            "rast_depth": rast_depth_pred,
            "valid_mask": valid_mask,
        }

    def diff_rasterization(self, posMap, full=False, debug=False, attr=None):
        if debug:
            st = time.time()

        B, img_h, img_w = posMap.shape[0], self.config["data.img_h"], self.config["data.img_w"]

        rast_out, rast_attr = self.renderer.render(
            self.M, posMap, self.tris, attr=attr
        )  # [B, img_h, img_w, 4], [B, img_h, img_w, 4]
        rast_out = rast_out[:, :img_h, :img_w]
        rast_depth = rast_attr[:, :img_h, :img_w, :1]  # [B, img_h, img_w, 1]
        rast_features = rast_attr[:, :img_h, :img_w, 1:]
        if not full:
            tri_ids = rast_out[:, :, :, 3].long()  # [B, img_h, img_w]
            valid = tri_ids > 0  # vaild pixels, [B, img_h, img_w]
        else:
            valid = torch.ones(self.photo.shape[:3]).bool().to(self.photo.device)

        if debug:
            timing(self.logger, st, label="[Rasterization]")

        return {"rast_out": rast_out, "rast_depth": rast_depth, "rast_features": rast_features, "valid_pixels": valid}

    def render_all_pixels(self, rast_res, dec_res, chunk=1024, debug=False):
        if debug:
            st = time.time()

        verts_w, feats = dec_res["verts_w"], dec_res["feats"]
        rast_depth, valid = rast_res["rast_depth"], rast_res["valid_pixels"]

        photo = torch.zeros_like(self.photo).to(self.photo.device)
        depth = torch.zeros_like(self.depth_map).to(self.depth_map.device)
        # v1: using for loop to process batch
        b = 0
        neural_points = NeuralPoints(self.config, verts_w[b], feats[b], is_val=True)
        # if self.config["rendering.point_dir_mode"]:
        #     neural_points.setNormals(v_normals)

        pixel_idx = self.pixel_idx[b][valid[b]][None]  # [1, N, 2]
        raydir = self.raydir[b][valid[b]][None]
        camrotc2w = self.camrotc2w[b : b + 1]
        campos = self.campos[b : b + 1]
        near = torch.tensor([self.config["rendering.near"]]).cuda()
        far = torch.tensor([self.config["rendering.far"]]).cuda()
        if self.config["rendering.sampling_method"] == "dcenter":
            sampling_center = self.get_sampling_center(
                rast_depth.clone(), pixel_idx, span=self.config["rendering.span"]
            )
        else:
            sampling_center = rast_depth[:, pixel_idx[0, :, 1].long(), pixel_idx[0, :, 0].long(), 0]
        if debug:
            it = timing(self.logger, st, label="[Build Neural Point Cloud]")

        total_pixel = pixel_idx.shape[1]
        for i in range(0, total_pixel, chunk):
            (
                sampled_color,
                sampled_Rw2c,
                sampled_dir,
                sampled_conf,
                sampled_embedding,
                sampled_xyz_pers,
                sampled_xyz,
                sample_pnt_mask,
                sample_loc,
                sample_loc_w,
                sample_ray_dirs,
                ray_mask_tensor,
                vsize,
                grid_vox_sz,
            ) = neural_points(
                {
                    "pixel_idx": pixel_idx[:, i : i + chunk],
                    "camrotc2w": camrotc2w,
                    "campos": campos,
                    "near": near,
                    "far": far,
                    "raydir": raydir[:, i : i + chunk],
                    "sampling_center": sampling_center[:, i : i + chunk],
                }
            )
            if debug and (i == 0):
                it = timing(self.logger, it, label="[Search KNN]")

            decoded_features, ray_valid, weight, conf_coefficient = self.models["aggregator"](
                sampled_color,
                sampled_Rw2c,
                sampled_dir,
                sampled_conf,
                sampled_embedding,
                sampled_xyz_pers,
                sampled_xyz,
                sample_pnt_mask,
                sample_loc,
                sample_loc_w,
                sample_ray_dirs,
                vsize,
                grid_vox_sz,
            )
            if debug and (i == 0):
                it = timing(self.logger, it, label="[Decode MLPs]")

            # point-nerf 计算 sample points 间距离
            # ray_dist = torch.cummax(sample_loc[..., 2], dim=-1)[0]
            # ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize[2], device=ray_dist.device)], dim=-1)

            # mask = ray_dist < 1e-8
            # if self.config["rendering.raydist_mode_unit"] > 0:
            #     mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
            # mask = mask.to(torch.float32)
            # ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
            # ray_dist *= ray_valid.float()

            # My Implementation
            z_vals = torch.cummax(sample_loc[..., 2], dim=-1)[0]  # [B, R, SR]
            ray_dist = torch.cat(
                [
                    z_vals[..., 1:] - z_vals[..., :-1],
                    torch.zeros((z_vals.shape[0], z_vals.shape[1], 1), device=z_vals.device),
                ],
                dim=-1,
            )

            last_valid_idx = torch.cumsum(ray_valid, dim=-1).argmax(-1)[0].long()  # [R]
            rays = torch.tensor(range(last_valid_idx.shape[0])).long().to(last_valid_idx.device)
            ray_dist[:, rays, last_valid_idx] = 1e10

            # mask = ray_dist < 1e-8
            # if self.config["rendering.raydist_mode_unit"] > 0:
            #     mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
            # mask = mask.to(torch.float32)
            # ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
            # ray_dist *= ray_valid.float()

            bg_color = None
            (ray_color, point_color, opacity, acc_transmission, blend_weight) = ray_march(
                ray_dist, ray_valid, decoded_features, render_func, blend_func, bg_color
            )
            ray_color = tone_map(ray_color)
            ray_depth = (z_vals * blend_weight[..., 0]).sum(-1)

            pidx = pixel_idx[0, i : i + chunk][ray_mask_tensor[0].bool()].long()

            photo[b, pidx[..., 1], pidx[..., 0].long()] = ray_color.data
            depth[b, pidx[..., 1], pidx[..., 0].long()] = ray_depth.data
            if debug and (i == 0):
                it = timing(self.logger, it, label="[Volume Rendering]")

        return {"photo": photo, "depth": depth}

    def eval_valid_ray(self, chunk=1024, full=False, debug=False):
        self.set_eval()

        if debug:
            st = time.time()
        enc_res = self.encode(self.avgTex, self.posMap, is_val=False, debug=debug)
        dec_res = self.net_decode(enc_res["z_code"], debug=debug)

        # nvdiffrast: rasterization
        posMap = dec_res["posMap"]
        rast_res = self.diff_rasterization(
            posMap.reshape(posMap.shape[0], 3, -1).permute((0, 2, 1)), full=full, debug=debug, attr=dec_res["down4feats"]
        )

        # neural points ray marching
        render_res = self.render_all_pixels(rast_res, dec_res, chunk=chunk, debug=debug)
        if debug:
            timing(self.logger, st, label="[Total Time]")

        return {
            "mean": enc_res["mean"],
            "logstd": enc_res["logstd"],
            "posMap": posMap,
            "dispMap": dec_res["dispMap"],
            "verts": dec_res["verts"],
            "render_rgb": render_res["photo"],
            "render_depth": render_res["depth"],
            "rast_depth": rast_res["rast_depth"][..., 0],
            "rast_features": rast_res["rast_features"],
            "valid_pixel": rast_res["valid_pixels"],
            "feats": dec_res["feats"]
        }

    def network_forward(self, is_val=False):
        enc_res = self.encode(self.avgTex, self.posMap)
        dec_res = self.decode(enc_res["z_code"])

        outputs = dec_res
        outputs["logstd"] = enc_res["logstd"]
        outputs["mean"] = enc_res["mean"]

        return outputs

    def compute_loss(self, outputs, is_val=False):
        logstd = outputs["logstd"]
        mean = outputs["mean"]
        posMap = outputs["posMap"]
        dispMap = outputs["dispMap"]
        render_rgb = outputs["render_rgb"]
        render_depth = outputs["render_depth"]
        rast_depth = outputs["rast_depth"]

        B = posMap.shape[0]
        # KL Loss
        kl_loss = 0.5 * torch.mean(
            torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd
        )  # kl loss 的推导可见 chrome 的 tutorial

        # RGB Loss
        rgb_loss_err = torch.linalg.norm(render_rgb - self.photo, dim=-1)
        rgb_loss = rgb_loss_err.mean()

        # LPIPS loss
        if is_val:
            pred_patch = render_rgb.permute((0, 3, 1, 2))
            gt_patch = self.photo.permute((0, 3, 1, 2))
            lpips_loss = self.lpips(pred_patch, gt_patch, normalize=True)
        else:
            if self.sample_method == "patch":
                p = self.config["rendering.random_sample_size"]
                pred_patch = render_rgb.permute((0, 2, 1)).reshape(B, 3, p, p)
                gt_patch = self.photo.permute((0, 2, 1)).reshape(B, 3, p, p)
                lpips_loss = self.lpips(pred_patch, gt_patch, normalize=True)
            else:
                lpips_loss = torch.tensor(0.0).cuda()

        # Render Depth Loss
        depth_mask = torch.abs(render_depth - self.depth_map) < self.config["training.depth_thres"]
        depth_loss = torch.abs((render_depth - self.depth_map) * depth_mask).mean()

        # Rasterized Depth Loss
        rast_depth_mask = torch.abs(rast_depth - self.depth_map) < self.config["training.depth_thres"]
        rast_depth_loss = torch.abs((rast_depth - self.depth_map) * rast_depth_mask).mean()

        # Mesh Loss
        coarseMesh = self.posMap * self.posStd + self.posMean
        self.verts = coarseMesh.reshape(B, 3, -1).permute((0, 2, 1))  # [B, 65536, 3]
        verts = posMap.reshape(B, 3, -1).permute((0, 2, 1))  # [B, 65536, 3]
        mesh_loss = torch.linalg.norm((verts - self.verts) * self.mouth_mask[None].expand(B, -1, -1), dim=-1).mean()

        # dispMap regularization
        if self.config["ablation.use_disp"]:
            disp_norm = torch.linalg.norm(dispMap, dim=1)
            disp_reg_loss = (disp_norm * (disp_norm > self.config["training.disp_thres"])).mean()
        else:
            disp_reg_loss = torch.tensor(0.0).cuda()

        # posMap grad smooth loss
        grad_smooth_loss = gradient_smooth(
            posMap, method=self.config["training.smooth_method"], beta=self.config["training.smooth_beta"]
        )

        # total loss
        lambda_rgb = update_lambda(
            self.config["training.lambda_rgb"],
            self.config["training.lambda_rgb_slope"],
            self.config["training.lambda_rgb_end"],
            self.global_step,
            self.config["training.lambda_interval"],
        )
        lambda_depth = update_lambda(
            self.config["training.lambda_depth"],
            self.config["training.lambda_depth_slope"],
            self.config["training.lambda_depth_end"],
            self.global_step,
            self.config["training.lambda_interval"],
        )
        lambda_mesh = update_lambda(
            self.config["training.lambda_mesh"],
            self.config["training.lambda_mesh_slope"],
            self.config["training.lambda_mesh_end"],
            self.global_step,
            self.config["training.lambda_interval"],
        )
        loss = (
            lambda_rgb * rgb_loss
            + lambda_depth * depth_loss
            + lambda_mesh * mesh_loss
            + self.config["training.lambda_sg"] * grad_smooth_loss
            + self.config["training.lambda_kl"] * kl_loss
            + self.config["training.lambda_lpips"] * lpips_loss
            + self.config["training.lambda_rdepth"] * rast_depth_loss
            + self.config["training.lambda_disp"] * disp_reg_loss
        )

        loss_dict = {
            "loss": loss,
            "loss_pho/rgb": rgb_loss,
            "loss_pho/lpips": lpips_loss,
            "loss_geo/depth": depth_loss,
            "loss_geo/rast_depth": rast_depth_loss,
            "loss_geo/mesh": mesh_loss,
            "loss_geo/posmap_grad": grad_smooth_loss,
            "loss_reg/kl_loss": kl_loss,
            "loss_reg/disp": disp_reg_loss,
            "rgb_loss_err": rgb_loss_err,
        }

        visualization_dict = {}

        return loss_dict, visualization_dict

    def log_training(self, epoch, step, global_step, dataset_length, loss_dict):
        loss = loss_dict["loss"]
        loss_rgb = loss_dict["loss_pho/rgb"]
        loss_lpips = loss_dict["loss_pho/lpips"]
        loss_depth = loss_dict["loss_geo/depth"]
        loss_rast_depth = loss_dict["loss_geo/rast_depth"]
        loss_mesh = loss_dict["loss_geo/mesh"]
        loss_sg = loss_dict["loss_geo/posmap_grad"]
        loss_kl = loss_dict["loss_reg/kl_loss"]
        loss_disp = loss_dict["loss_reg/disp"]

        lr = self.scheduler.get_last_lr()[0]
        self.logger.info(
            "epoch [%.3d] step [%d/%d] global_step = %d loss = %.4f lr = %.6f\n"
            "        rgb = %.4f\n"
            "        lpips = %.4f\n"
            "        depth = %.4f\n"
            "        rast_depth = %.4f\n"
            "        mesh = %.4f\n"
            "        grad_smooth = %.4f\n"
            "        disp = %.4f\n"
            "        kl = %.4f\n"
            % (
                epoch,
                step,
                dataset_length,
                self.global_step,
                loss.item(),
                lr,
                loss_rgb.item(),
                loss_lpips.item(),
                loss_depth.item(),
                loss_rast_depth.item(),
                loss_mesh.item(),
                loss_sg.item(),
                loss_disp.item(),
                loss_kl.item(),
            )
        )

        # Write losses to tensorboard
        # Update avg meters
        for key, value in self.train_losses.items():
            if self.tb_writer:
                self.tb_writer.add_scalar(key, loss_dict[key].item(), global_step)
            value.update(loss_dict[key].item())

    def run_eval(self, val_loader):
        if self.config["global_rank"] == 0:
            self.logger.info("Start running evaluation on validation set:")
        self.set_eval()

        # clear train losses average meter
        for val_loss_item in self.val_losses.values():
            val_loss_item.reset()

        batch_count = 0
        with torch.no_grad():
            for step, items in enumerate(val_loader):
                batch_count += 1
                if self.config["global_rank"] == 0 and batch_count % 20 == 0:
                    self.logger.info("    Eval progress: {}/{}".format(batch_count, len(val_loader)))

                self.set_data(items)
                outputs = self.eval_valid_ray(chunk=self.config["rendering.random_sample_size"] ** 2)
                loss_dict, visualization_dict = self.compute_loss(outputs, is_val=True)

                for key in loss_dict.keys():
                    dist.all_reduce(loss_dict[key], op=dist.ReduceOp.SUM)
                    loss_dict[key] /= self.config["world_size"]

                valid = outputs["valid_pixel"] * items["mask"].cuda()
                photo, photo_gt = outputs["render_rgb"][valid] * 255, self.photo[valid] * 255
                m_mse = ((photo - photo_gt) ** 2).mean()
                psnr = 10 * torch.log10(65025 / m_mse)
                dist.all_reduce(m_mse, op=dist.ReduceOp.SUM)
                dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
                m_mse /= self.config["world_size"]
                psnr /= self.config["world_size"]

                loss_dict["loss_pho/mse"] = m_mse
                loss_dict["loss_pho/psnr"] = psnr

                self.log_val(step, loss_dict, visualization_dict)

            # log evaluation result
            if self.config["global_rank"] == 0:
                self.logger.info("Evaluation finished, average losses: ")
                for v in self.val_losses.values():
                    self.logger.info("    {}".format(v))

                # Write val losses to tensorboard
                if self.tb_writer:
                    for key, value in self.val_losses.items():
                        self.tb_writer.add_scalar(key + "//val", value.avg, self.global_step)

        self.set_train()

    def log_val(self, step, loss_dict, visualization_dict):
        B = self.avgTex.shape[0]
        # loss logging
        for key, value in self.val_losses.items():
            value.update(loss_dict[key].item(), n=B * self.config["world_size"])

        # TODO: 如果需要，可以将一些渲染结果存在visualization_dict中，然后用tb_writer保存一些结果图用于可视化

    def train_epoch(self, train_loader, val_loader, show_time=False):
        if hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(self.current_epoch)

        # convert models to traning mode
        self.set_train()

        if show_time:
            batch_start = time.time()

        for step, items in enumerate(train_loader):
            # print(str(self.config["global_rank"]) + ":" + items["exp"][0] + ":" + items["frame"][0] + ":" + items["cam_idx"][0])

            step += 1

            if show_time:
                load_data_end = timing(self.logger, batch_start, "Data Loading End")

            self.global_step += 1
            self.set_data(items)

            if show_time:
                set_data_end = timing(self.logger, load_data_end, "Data Setting End")

            outputs = self.network_forward()

            if "posMap" not in outputs.keys():
                continue

            if show_time:
                network_end = timing(self.logger, set_data_end, "Network End")

            loss_dict, _ = self.compute_loss(outputs)
            with torch.no_grad():
                train_loader.dataset.update_errmap(
                    self.idx[0], self.pixel_idx[0], outputs["valid_mask"], loss_dict["rgb_loss_err"][0].clone()
                )

            if show_time:
                compute_loss_end = timing(self.logger, network_end, "Loss Computing End")

            loss = loss_dict["loss"]

            # DEBUG: save the checkpoint before NaN Loss
            # if loss.isnan().any():
            #     if self.config["global_rank"] == 0:
            #         checkpoint_path = os.path.join(self.config["local_workspace"], "nan_break.pth")
            #         save_dict = {"optimizer": self.optimizer.state_dict()}
            #         for k, m in self.models.items():
            #             save_dict[k] = m.state_dict()

            #         torch.save(save_dict, checkpoint_path)
            #         self.logger.info("NaN break checkpoint has saved at {}".format(checkpoint_path))
            #     return False

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters_to_train, 0.01)
            # plot_grad_flow(self.models['pixelDec'].module.xyz_enc.named_parameters())
            self.optimizer.step()

            if show_time:
                optimize_end = timing(self.logger, compute_loss_end, "Optimizing End")

            # logging
            if step > 0 and (step % 10 == 0 or step == len(train_loader)) and self.config["global_rank"] == 0:
                self.log_training(self.current_epoch, step, self.global_step, len(train_loader), loss_dict)

            if step > 0 and self.global_step % 5000 == 0 and self.config["global_rank"] == 0:
                # Save model and put checkpoint to hdfs
                checkpoint_path = os.path.join(self.config["local_workspace"], "checkpoint_latest.pth")
                save_dict = {
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                    "lossmap": train_loader.dataset.errMap,
                }
                for k, m in self.models.items():
                    save_dict[k] = m.state_dict()

                torch.save(save_dict, checkpoint_path)
                self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

            # if step > 0 and self.global_step % 10000 == 0 and self.config["global_rank"] == 0:
            #     # save loss map
            #     save_path = os.path.join(self.config["local_workspace"], "loss_map{}.npy".format(self.global_step))
            #     np.save(save_path, train_loader.dataset.errMap)
            #     save_path = os.path.join(self.config["local_workspace"], "frame_list.pth")
            #     torch.save(train_loader.dataset.framelist, save_path)

            #     self.logger.info("Loss map saved at {}".format(save_path))

            if self.global_step > 0 and self.global_step % self.config["training.eval_interval"] == 0:
                self.run_eval(val_loader)

                # Save model
                if self.config["global_rank"] == 0:
                    checkpoint_path = os.path.join(
                        self.config["local_workspace"], "checkpoint_%012d.pth" % self.global_step
                    )

                    save_dict = {
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": self.current_epoch,
                        "global_step": self.global_step,
                        "lossmap": train_loader.dataset.errMap,
                    }
                    for k, m in self.models.items():
                        save_dict[k] = m.state_dict()

                    torch.save(save_dict, checkpoint_path)

            # if self.config["global_rank"] == 0:
            #     free, total = torch.cuda.mem_get_info()
            #     self.logger.info('free and total mem: {}GB / {}GB'.format(free/1024/1024/1024, total/1024/1024/1024))

            if show_time:
                batch_start = timing(self.logger, batch_start, "Iteration Time")
                self.logger.info("==============================")

        return True
