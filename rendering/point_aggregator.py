import torch
import torch.nn as nn

import numpy as np

from .render_utils import init_seq, positional_encoding
from .layers import Siren


class PointAggregator(nn.Module):
    def __init__(self, config):
        super(PointAggregator, self).__init__()
        self.act = getattr(nn, config["rendering.act_type"], None)
        self.config = config

        self.dist_dim = (
            (4 if config["rendering.agg_dist_pers"] == 30 else 6) if config["rendering.agg_dist_pers"] > 9 else 3
        )
        self.dist_func = getattr(self, config["rendering.agg_distance_kernel"], None)
        assert self.dist_func is not None, "InterpAggregator doesn't have disance_kernel {} ".format(
            config["rendering.agg_distance_kernel"]
        )
        self.axis_weight = (
            None
            if config["rendering.agg_axis_weight"] is None
            else torch.as_tensor(config["rendering.agg_axis_weight"], dtype=torch.float32, device="cuda")[
                None, None, None, None, :
            ]
        )

        self.num_freqs = config["rendering.num_pos_freqs"] if config["rendering.num_pos_freqs"] > 0 else 0
        self.num_viewdir_freqs = (
            config["rendering.num_viewdir_freqs"] if config["rendering.num_viewdir_freqs"] > 0 else 0
        )

        self.pnt_channels = (2 * self.num_freqs * 3) if self.num_freqs > 0 else 3
        self.viewdir_channels = (
            (2 * self.num_viewdir_freqs * 3 + config["rendering.view_ori"] * 3) if self.num_viewdir_freqs > 0 else 3
        )

        self.which_agg_model = (
            config["rendering.which_agg_model"].split("_")[0]
            if config["rendering.which_agg_model"].startswith("feathyper")
            else config["rendering.which_agg_model"]
        )
        getattr(self, self.which_agg_model + "_init", None)(config)

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        self.color_act = torch.nn.Sigmoid()

    def fullmlp_init(self, config):
        block_init_lst = []
        dist_xyz_dim = (
            self.dist_dim
            if config["rendering.dist_xyz_freq"] == 0
            else 2 * abs(config["rendering.dist_xyz_freq"]) * self.dist_dim
        )
        in_channels = (
            config["rendering.point_features_dim"]
            + (0 if config["rendering.agg_feat_xyz_mode"] == "None" else self.pnt_channels)
            - (
                config["rendering.weight_feat_dim"]
                if config["rendering.agg_distance_kernel"] in ["feat_intrp", "meta_intrp"]
                else 0
            )
            - (config["rendering.sh_degree"] ** 2 if config["rendering.agg_distance_kernel"] == "sh_intrp" else 0)
            - (7 if config["rendering.agg_distance_kernel"] == "gau_intrp" else 0)
        )
        in_channels += (
            2 * config["rendering.num_feat_freqs"] * in_channels if config["rendering.num_feat_freqs"] > 0 else 0
        ) + (dist_xyz_dim if config["rendering.agg_intrp_order"] > 0 else 0)

        if config["rendering.shading_feature_mlp_layer1"] > 0:
            out_channels = config["rendering.shading_feature_num"]
            block1 = []
            for i in range(config["rendering.shading_feature_mlp_layer1"]):
                block1.append(nn.Linear(in_channels, out_channels))
                block1.append(self.act(0.2, inplace=True))
                in_channels = out_channels
            self.block1 = nn.Sequential(*block1)
            block_init_lst.append(self.block1)
        else:
            self.block1 = self.passfunc

        if config["rendering.shading_feature_mlp_layer2"] > 0:
            in_channels = (
                in_channels
                + (0 if config["rendering.agg_feat_xyz_mode"] == "None" else self.pnt_channels)
                + (
                    dist_xyz_dim
                    if (config["rendering.agg_intrp_order"] > 0 and config["rendering.num_feat_freqs"] == 0)
                    else 0
                )
            )
            out_channels = config["rendering.shading_feature_num"]
            block2 = []
            for i in range(config["rendering.shading_feature_mlp_layer2"]):
                block2.append(nn.Linear(in_channels, out_channels))
                block2.append(self.act(0.2, inplace=True))
                in_channels = out_channels
            self.block2 = nn.Sequential(*block2)
            block_init_lst.append(self.block2)
        else:
            self.block2 = self.passfunc

        if config["rendering.shading_feature_mlp_layer3"] > 0:
            in_channels = (
                in_channels
                + (3 if config["rendering.point_color_mode"] else 0)
                + (4 if config["rendering.point_dir_mode"] else 0)
            )
            out_channels = config["rendering.shading_feature_num"]
            block3 = []
            for i in range(config["rendering.shading_feature_mlp_layer3"]):
                block3.append(nn.Linear(in_channels, out_channels))
                block3.append(self.act(0.2, inplace=True))
                in_channels = out_channels
            self.block3 = nn.Sequential(*block3)
            block_init_lst.append(self.block3)
        else:
            self.block3 = self.passfunc

        alpha_block = []
        in_channels = config["rendering.shading_feature_num"] + (
            0 if config["rendering.agg_alpha_xyz_mode"] == "None" else self.pnt_channels
        )
        out_channels = int(config["rendering.shading_feature_num"] / 2)
        for i in range(config["rendering.shading_alpha_mlp_layer"] - 1):
            alpha_block.append(nn.Linear(in_channels, out_channels))
            alpha_block.append(self.act(0.2, inplace=False))
            in_channels = out_channels
        alpha_block.append(nn.Linear(in_channels, 1))
        self.alpha_branch = nn.Sequential(*alpha_block)
        block_init_lst.append(self.alpha_branch)

        color_block = []
        in_channels = (
            config["rendering.shading_feature_num"]
            + self.viewdir_channels
            + (0 if config["rendering.agg_color_xyz_mode"] == "None" else self.pnt_channels)
        )
        out_channels = int(config["rendering.shading_feature_num"] / 2)
        for i in range(config["rendering.shading_color_mlp_layer"] - 1):
            color_block.append(nn.Linear(in_channels, out_channels))
            color_block.append(self.act(0.2, inplace=True))
            in_channels = out_channels
        color_block.append(nn.Linear(in_channels, 3))
        self.color_branch = nn.Sequential(*color_block)
        block_init_lst.append(self.color_branch)

        for m in block_init_lst:
            init_seq(m)

    def viewmlp_init(self, config):
        block_init_lst = []
        dist_xyz_dim = (
            self.dist_dim
            if config["rendering.dist_xyz_freq"] == 0
            else 2 * abs(config["rendering.dist_xyz_freq"]) * self.dist_dim
        )
        feats_channels = (
            config["rendering.point_features_dim"]
            + (0 if config["rendering.agg_feat_xyz_mode"] == "None" else self.pnt_channels)
            - (
                config["rendering.weight_feat_dim"]
                if config["rendering.agg_distance_kernel"] in ["feat_intrp", "meta_intrp"]
                else 0
            )
            - (config["rendering.sh_degree"] ** 2 if config["rendering.agg_distance_kernel"] == "sh_intrp" else 0)
            - (7 if config["rendering.agg_distance_kernel"] == "gau_intrp" else 0)
        )
        feats_channels += dist_xyz_dim + (
            2 * config["rendering.num_feat_freqs"] * feats_channels if config["rendering.num_feat_freqs"] > 0 else 0
        )

        alpha_block = []
        in_channels = feats_channels + 0 if config["rendering.agg_alpha_xyz_mode"] == "None" else self.pnt_channels
        out_channels = int(config["rendering.shading_feature_num"] / 2)
        for i in range(config["rendering.shading_alpha_mlp_layer"] - 1):
            alpha_block.append(nn.Linear(in_channels, out_channels))
            alpha_block.append(self.act(0.2, inplace=True))
            in_channels = out_channels
        alpha_block.append(nn.Linear(in_channels, 1))
        self.alpha_branch = nn.Sequential(*alpha_block)
        block_init_lst.append(self.alpha_branch)

        color_block = []
        in_channels = (
            feats_channels
            + self.viewdir_channels
            + (0 if config["rendering.agg_color_xyz_mode"] == "None" else self.pnt_channels)
        )
        out_channels = int(config["rendering.shading_feature_num"] / 2)
        for i in range(config["rendering.shading_color_mlp_layer"] - 1):
            color_block.append(nn.Linear(in_channels, out_channels))
            color_block.append(self.act(0.2, inplace=True))
            in_channels = out_channels
        color_block.append(nn.Linear(in_channels, 3))
        self.color_branch = nn.Sequential(*color_block)
        block_init_lst.append(self.color_branch)

        for m in block_init_lst:
            init_seq(m)

    def raw2out_density(self, raw_density):
        if self.config["rendering.act_super"] > 0:
            # return self.density_act(raw_density - 1)  # according to mip nerf, to stablelize the training
            return self.density_super_act(raw_density - 1)  # according to mip nerf, to stablelize the training
        else:
            return self.density_act(raw_density)

    def raw2out_color(self, raw_color):
        color = self.color_act(raw_color)
        if self.config["rendering.act_super"] > 0:
            color = color * (1 + 2 * 0.001) - 0.001  # according to mip nerf, to stablelize the training
        return color

    def fullmlp(
        self,
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
        weight,
        pnt_mask_flat,
        pts,
        viewdirs,
        total_len,
        ray_valid,
        in_shape,
        dists,
    ):
        B, R, SR, K, _ = dists.shape
        sampled_Rw2c = sampled_Rw2c.transpose(-1, -2)
        uni_w2c = sampled_Rw2c.dim() == 2
        if not uni_w2c:
            sampled_Rw2c_ray = sampled_Rw2c[:, :, :, 0, :, :].view(-1, 3, 3)
            sampled_Rw2c = sampled_Rw2c.reshape(-1, 3, 3)[pnt_mask_flat, :, :]
        pts_ray, pts_pnt = None, None
        if (
            self.config["rendering.agg_feat_xyz_mode"] != "None"
            or self.config["rendering.agg_alpha_xyz_mode"] != "None"
            or self.config["rendering.agg_color_xyz_mode"] != "None"
        ):
            if self.num_freqs > 0:
                pts = positional_encoding(pts, self.num_freqs)
            pts_ray = pts[ray_valid, :]
            if self.config["rendering.agg_feat_xyz_mode"] != "None" and self.config["rendering.agg_intrp_order"] > 0:
                pts_pnt = pts[..., None, :].repeat(1, K, 1).view(-1, pts.shape[-1])
                if self.config["rendering.apply_pnt_mask"] > 0:
                    pts_pnt = pts_pnt[pnt_mask_flat, :]
        viewdirs = viewdirs @ sampled_Rw2c if uni_w2c else (viewdirs[..., None, :] @ sampled_Rw2c_ray).squeeze(-2)
        if self.num_viewdir_freqs > 0:
            viewdirs = positional_encoding(viewdirs, self.num_viewdir_freqs, ori=True)
            ori_viewdirs, viewdirs = viewdirs[..., :3], viewdirs[..., 3:]
        viewdirs = viewdirs[ray_valid, :]

        if self.config["rendering.agg_intrp_order"] == 0:
            feat = torch.sum(sampled_embedding * weight[..., None], dim=-2)
            feat = feat.view([-1, feat.shape[-1]])[ray_valid, :]
            if self.config["rendering.num_feat_freqs"] > 0:
                feat = torch.cat([feat, positional_encoding(feat, self.config["rendering.num_feat_freqs"])], dim=-1)
            pts = pts_ray
        else:
            dists_flat = dists.view(-1, dists.shape[-1])
            if self.config["rendering.apply_pnt_mask"] > 0:
                dists_flat = dists_flat[pnt_mask_flat, :]
            dists_flat /= (
                1.0
                if self.config["rendering.dist_xyz_deno"] == 0.0
                else float(self.config["rendering.dist_xyz_deno"] * np.linalg.norm(vsize))
            )
            dists_flat[..., :3] = (
                dists_flat[..., :3] @ sampled_Rw2c
                if uni_w2c
                else (dists_flat[..., None, :3] @ sampled_Rw2c).squeeze(-2)
            )
            if self.config["rendering.dist_xyz_freq"] != 0:
                # print(dists.dtype, (self.opt.dist_xyz_deno * np.linalg.norm(vsize)).dtype, dists_flat.dtype)
                dists_flat = positional_encoding(dists_flat, self.config["rendering.dist_xyz_freq"])
            feat = sampled_embedding.view(-1, sampled_embedding.shape[-1])
            # print("feat", feat.shape)

            if self.config["rendering.apply_pnt_mask"] > 0:
                feat = feat[pnt_mask_flat, :]
            if self.config["rendering.num_feat_freqs"] > 0:
                feat = torch.cat([feat, positional_encoding(feat, self.config["rendering.num_feat_freqs"])], dim=-1)
            feat = torch.cat([feat, dists_flat], dim=-1)  # channel 284
            weight = weight.view(B * R * SR, K, 1)
            pts = pts_pnt

        if self.config["rendering.agg_feat_xyz_mode"] != "None":
            feat = torch.cat([feat, pts], dim=-1)
        # print("feat",feat.shape) # 501
        feat = self.block1(feat)

        if self.config["rendering.shading_feature_mlp_layer2"] > 0:
            if self.opt.agg_feat_xyz_mode != "None":
                feat = torch.cat([feat, pts], dim=-1)
            if self.opt.agg_intrp_order > 0:
                feat = torch.cat([feat, dists_flat], dim=-1)
            feat = self.block2(feat)

        if self.config["rendering.shading_feature_mlp_layer3"] > 0:
            if sampled_color is not None:
                sampled_color = sampled_color.view(-1, sampled_color.shape[-1])
                if self.config["rendering.apply_pnt_mask"] > 0:
                    sampled_color = sampled_color[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_color], dim=-1)
            if sampled_dir is not None:
                sampled_dir = sampled_dir.view(-1, sampled_dir.shape[-1])
                if self.config["rendering.apply_pnt_mask"] > 0:
                    sampled_dir = sampled_dir[pnt_mask_flat, :]
                    sampled_dir = (
                        sampled_dir @ sampled_Rw2c
                        if uni_w2c
                        else (sampled_dir[..., None, :] @ sampled_Rw2c).squeeze(-2)
                    )
                ori_viewdirs = ori_viewdirs[..., None, :].repeat(1, K, 1).view(-1, ori_viewdirs.shape[-1])
                if self.config["rendering.apply_pnt_mask"] > 0:
                    ori_viewdirs = ori_viewdirs[pnt_mask_flat, :]
                feat = torch.cat(
                    [feat, sampled_dir - ori_viewdirs, torch.sum(sampled_dir * ori_viewdirs, dim=-1, keepdim=True)],
                    dim=-1,
                )
            feat = self.block3(feat)

        if self.config["rendering.agg_intrp_order"] == 1:
            if self.config["rendering.apply_pnt_mask"] > 0:
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
            feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]

            alpha_in = feat
            if self.config["rendering.agg_alpha_xyz_mode"] != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)

            alpha = self.raw2out_density(self.alpha_branch(alpha_in))

            color_in = feat
            if self.config["rendering.agg_color_xyz_mode"] != "None":
                color_in = torch.cat([color_in, pts], dim=-1)

            color_in = torch.cat([color_in, viewdirs], dim=-1)
            color_output = self.raw2out_color(self.color_branch(color_in))

            # print("color_output", torch.sum(color_output), color_output.grad)

            output = torch.cat([alpha, color_output], dim=-1)

        elif self.config["rendering.agg_intrp_order"] == 2:
            alpha_in = feat
            if self.config["rendering.agg_alpha_xyz_mode"] != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)
            alpha = self.raw2out_density(self.alpha_branch(alpha_in))
            # print(alpha_in.shape, alpha_in)

            if self.config["rendering.apply_pnt_mask"] > 0:
                alpha_holder = torch.zeros([B * R * SR * K, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
                alpha_holder[pnt_mask_flat, :] = alpha
            else:
                alpha_holder = alpha
            alpha = alpha_holder.view(B * R * SR, K, alpha_holder.shape[-1])
            alpha = torch.sum(alpha * weight, dim=-2).view([-1, alpha.shape[-1]])[ray_valid, :]  # alpha:

            if self.config["rendering.apply_pnt_mask"] > 0:
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
            feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]

            color_in = feat
            if self.config["rendering.agg_color_xyz_mode"] != "None":
                color_in = torch.cat([color_in, pts], dim=-1)

            color_in = torch.cat([color_in, viewdirs], dim=-1)
            color_output = self.raw2out_color(self.color_branch(color_in))

            output = torch.cat([alpha, color_output], dim=-1)

            # print("output_placeholder", output_placeholder.shape)

        output_placeholder = torch.zeros(
            [total_len, self.config["rendering.shading_color_channel_num"] + 1],
            dtype=torch.float32,
            device=output.device,
        )
        output_placeholder[ray_valid] = output
        return output_placeholder, None

    def viewmlp(
        self,
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
        weight,
        pnt_mask_flat,
        pts,
        viewdirs,
        total_len,
        ray_valid,
        in_shape,
        dists,
    ):
        B, R, SR, K, _ = dists.shape
        pts_ray, pts_pnt = None, None
        if (
            self.config["rendering.agg_feat_xyz_mode"] != "None"
            or self.config["rendering.agg_alpha_xyz_mode"] != "None"
            or self.config["rendering.agg_color_xyz_mode"] != "None"
        ):
            if self.num_freqs > 0:
                pts = positional_encoding(pts, self.num_freqs)
            pts_ray = pts[ray_valid, :]
            if self.config["rendering.agg_feat_xyz_mode"] != "None" and self.config["rendering.agg_intrp_order"] > 0:
                pts_pnt = pts[..., None, :].repeat(1, K, 1).view(-1, pts.shape[-1])
                if self.config["rendering.apply_pnt_mask"] > 0:
                    pts_pnt = pts_pnt[pnt_mask_flat, :]
        if self.num_viewdir_freqs > 0:
            viewdirs = positional_encoding(viewdirs, self.num_viewdir_freqs, ori=True)
            ori_viewdirs, viewdirs = viewdirs[..., :3], viewdirs[..., 3:]
        viewdirs = viewdirs[ray_valid, :]

        agg_embedding = torch.sum(sampled_embedding * weight[..., None], dim=-2)
        agg_embedding = agg_embedding.view(-1, agg_embedding.shape[-1])[ray_valid, :]
        if self.config["rendering.num_feat_freqs"] > 0:
            agg_embedding = torch.cat(
                [agg_embedding, positional_encoding(agg_embedding, self.config["rendering.num_feat_freqs"])], dim=-1
            )
        # pe_embedding = sampled_embedding
        # if self.config["rendering.num_feat_freqs"] > 0:
        #     pe_embedding = torch.cat(
        #         [pe_embedding, positional_encoding(pe_embedding, self.config["rendering.num_feat_freqs"])], dim=-1
        # )
        # agg_embedding = torch.sum(pe_embedding * weight[..., None], dim=-2)
        # agg_embedding = agg_embedding.view(-1, agg_embedding.shape[-1])[ray_valid, :]

        agg_dists = torch.sum(dists * weight[..., None], dim=-2)
        # if not requiring gradient, can not register hook
        # agg_dists.register_hook(lambda grad: print('[HOOK] agg_dists.grad.max: {}'.format(grad.max())))
        agg_dists = agg_dists.view(-1, agg_dists.shape[-1])[ray_valid, :]
        if self.config["rendering.dist_xyz_freq"] > 0:
            agg_dists = positional_encoding(agg_dists, self.config["rendering.dist_xyz_freq"])

        # pe_dists = dists.detach()
        # if self.config["rendering.dist_xyz_freq"] > 0:
        #     pe_dists = positional_encoding(pe_dists, self.config["rendering.dist_xyz_freq"])
        # agg_dists = torch.sum(pe_dists * weight[..., None], dim=-2)
        # agg_dists = agg_dists.view(-1, agg_dists.shape[-1])[ray_valid, :]

        feat = torch.cat([agg_embedding, agg_dists], dim=-1)
        pts = pts_ray
        
        alpha_in = feat
        if self.config["rendering.agg_alpha_xyz_mode"] != "None":
            alpha_in = torch.cat([alpha_in, pts], dim=-1)

        alpha = self.raw2out_density(self.alpha_branch(alpha_in))

        color_in = feat
        if self.config["rendering.agg_color_xyz_mode"] != "None":
            color_in = torch.cat([color_in, pts], dim=-1)

        color_in = torch.cat([color_in, viewdirs], dim=-1)
        color_output = self.raw2out_color(self.color_branch(color_in))

        output = torch.cat([alpha, color_output], dim=-1)

        output_placeholder = torch.zeros(
            [total_len, self.config["rendering.shading_color_channel_num"] + 1],
            dtype=torch.float32,
            device=output.device,
        )
        output_placeholder[ray_valid] = output
        return output_placeholder, None

    def passfunc(self, input):
        return input

    def linear(self, dists, pnt_mask, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] == 1):
            weights = 1.0 / torch.clamp(torch.norm(dists[..., :3].clone(), dim=-1), min=1e-6)
        else:
            weights = 1.0 / torch.clamp(
                torch.sqrt(torch.sum(torch.square(dists[..., :2]), dim=-1)) * axis_weight[..., 0]
                + torch.abs(dists[..., 2]) * axis_weight[..., 1],
                min=1e-6,
            )
        weights = pnt_mask * weights
        return weights

    def gradiant_clamp(self, sampled_conf, min=0.0001, max=1):
        diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
        return sampled_conf - diff.detach()

    def forward(
        self,
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
    ):
        # return B * R * SR * channel
        """
        :param sampled_conf: B x valid R x SR x K x 1
        :param sampled_embedding: B x valid R x SR x K x F
        :param sampled_xyz_pers:  B x valid R x SR x K x 3
        :param sampled_xyz:       B x valid R x SR x K x 3
        :param sample_pnt_mask:   B x valid R x SR x K
        :param sample_loc:        B x valid R x SR x 3
        :param sample_loc_w:      B x valid R x SR x 3
        :param sample_ray_dirs:   B x valid R x SR x 3
        :param vsize:
        :return:
        """
        ray_valid = torch.any(sample_pnt_mask, dim=-1).view(-1)
        total_len = len(ray_valid)
        in_shape = sample_loc_w.shape
        if total_len == 0 or torch.sum(ray_valid) == 0:
            # print("skip since no valid ray, total_len:", total_len, torch.sum(ray_valid))
            return (
                torch.zeros(
                    in_shape[:-1] + (self.config["rendering.shading_color_channel_num"] + 1,),
                    device=ray_valid.device,
                    dtype=torch.float32,
                ),
                ray_valid.view(in_shape[:-1]),
                None,
                None,
            )

        if self.config["rendering.agg_dist_pers"] < 0:
            dists = sample_loc_w[..., None, :]
        elif self.config["rendering.agg_dist_pers"] == 0:
            dists = sampled_xyz - sample_loc_w[..., None, :]
        elif self.config["rendering.agg_dist_pers"] == 1:
            dists = sampled_xyz_pers - sample_loc[..., None, :]
        elif self.config["rendering.agg_dist_pers"] == 2:
            if sampled_xyz_pers.shape[1] > 0:
                xdist = (
                    sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2]
                    - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                )
                ydist = (
                    sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2]
                    - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                )
                zdist = sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 3], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.config["rendering.agg_dist_pers"] == 10:
            if sampled_xyz_pers.shape[1] > 0:
                dists = sampled_xyz_pers - sample_loc[..., None, :]
                dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.config["rendering.agg_dist_pers"] == 20:
            if sampled_xyz_pers.shape[1] > 0:
                xdist = (
                    sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2]
                    - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                )
                ydist = (
                    sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2]
                    - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                )
                zdist = sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)

                # dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
                dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.config["rendering.agg_dist_pers"] == 30:
            if sampled_xyz_pers.shape[1] > 0:
                w_dists = sampled_xyz - sample_loc_w[..., None, :]
                dists = torch.cat(
                    [torch.sum(w_dists * sample_ray_dirs[..., None, :], dim=-1, keepdim=True), dists], dim=-1
                )
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 4], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)
        else:
            print("illegal agg_dist_pers code: ", self.config["rendering.agg_dist_pers"])
            exit()
        # self.print_point(dists, sample_loc_w, sampled_xyz, sample_loc, sampled_xyz_pers, sample_pnt_mask)

        # dists.register_hook(lambda grad: print('[HOOK] dists.grad.max: {}'.format(grad.max())))
        weight = self.dist_func(dists, sample_pnt_mask, axis_weight=self.axis_weight)

        if (
            self.config["rendering.agg_weight_norm"] > 0
            and self.config["rendering.agg_distance_kernel"] != "trilinear"
            and not self.config["rendering.agg_distance_kernel"].startswith("num")
        ):
            weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)

        pnt_mask_flat = sample_pnt_mask.view(-1)
        pts = sample_loc_w.view(-1, sample_loc_w.shape[-1])
        viewdirs = sample_ray_dirs.view(-1, sample_ray_dirs.shape[-1])
        conf_coefficient = 1
        if sampled_conf is not None:
            conf_coefficient = self.gradiant_clamp(sampled_conf[..., 0], min=0.0001, max=1)

        output, _ = getattr(self, self.which_agg_model, None)(
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
            weight * conf_coefficient,
            pnt_mask_flat,
            pts,
            viewdirs,
            total_len,
            ray_valid,
            in_shape,
            dists,
        )

        if (
            (self.config["rendering.sparse_loss_weight"] <= 0)
            and ("conf_coefficient" not in self.config["rendering.zero_one_loss_items"])
            and self.config["rendering.prob"] == 0
        ):
            weight, conf_coefficient = None, None

        return (
            output.view(in_shape[:-1] + (self.config["rendering.shading_color_channel_num"] + 1,)),
            ray_valid.view(in_shape[:-1]),
            weight,
            conf_coefficient,
        )
