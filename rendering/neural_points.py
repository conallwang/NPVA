import torch
import torch.nn as nn

from .point_query import lighting_fast_querier as lighting_fast_querier_cuda


class NeuralPoints(nn.Module):
    def __init__(self, config, verts, feats, conf=None, normals=None, is_val=False):
        """init method

        Args:
            config (dict): config options
            verts (float): BxNx3, points
            feats (float): BxNxN_f, point features
            conf (float, 0~1): N, point confidence
        """
        super().__init__()

        self.grid_vox_sz = 0
        self.points_conf, self.points_dir, self.points_color, self.eulers = None, None, None, None
        self.Rw2c = torch.eye(3, device=verts.device, dtype=verts.dtype)

        self.config = config
        self.xyz = verts
        self.points_embeding = feats

        # self.config["neural_points.query_size"] = self.config["neural_points.kernel_size"] if self.config["query_size"][0] == 0 else self.config["neural_points.query_size"]

        self.querier = lighting_fast_querier_cuda(config, is_val)

    def setNormals(self, normals):
        self.points_dir = normals

    def setConfidence(self, conf):
        self.points_conf = conf

    def w2pers(self, point_xyz, camrotc2w, campos):
        """world coordinates 2 camera coordinates

        Args:
            point_xyz (float): BxNx3, world coordinates of points
            camrot (float): Bx3x3, rotation matrix (c22)
            campos (float): Bx3, transition vector (c2w)
        """
        point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
        # 相当于 camrotc2w 的转置 @ (x - t)，xyz 为 camera 坐标系下坐标
        xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
        # print(xyz.shape, (point_xyz_shift[:, None, :] * camrot.T).shape)
        xper = xyz[:, :, 0] / (xyz[:, :, 2] + 1e-9)
        yper = xyz[:, :, 1] / (xyz[:, :, 2] + 1e-9)
        return torch.stack([xper, yper, xyz[:, :, 2]], dim=-1)

    def get_point_indices(self, inputs, cam_rot_tensor, cam_pos_tensor, pixel_idx_tensor, near_plane, far_plane):
        point_xyz_pers_tensor = self.w2pers(self.xyz, cam_rot_tensor, cam_pos_tensor)
        actual_numpoints_tensor = (
            torch.ones([point_xyz_pers_tensor.shape[0]], device=point_xyz_pers_tensor.device, dtype=torch.int32)
            * point_xyz_pers_tensor.shape[1]
        )
        # print("pixel_idx_tensor", pixel_idx_tensor)
        # print("point_xyz_pers_tensor", point_xyz_pers_tensor.shape)
        # print("actual_numpoints_tensor", actual_numpoints_tensor.shape)
        # sample_pidx_tensor: B, R, SR, K
        ray_dirs_tensor, sampling_center = inputs["raydir"], inputs["sampling_center"]
        # print("ray_dirs_tensor", ray_dirs_tensor.shape, self.xyz.shape)
        (
            sample_pidx_tensor,
            sample_loc_tensor,
            sample_loc_w_tensor,
            sample_ray_dirs_tensor,
            ray_mask_tensor,
            vsize,
            ranges,
        ) = self.querier.query_points(
            pixel_idx_tensor,
            self.xyz[None, ...],
            actual_numpoints_tensor,
            near_plane,
            far_plane,
            ray_dirs_tensor,
            cam_pos_tensor,
            cam_rot_tensor,
            sampling_center,
        )

        return (
            sample_pidx_tensor,
            sample_loc_tensor,
            ray_mask_tensor,
            point_xyz_pers_tensor,
            sample_loc_w_tensor,
            sample_ray_dirs_tensor,
            vsize,
        )

    def forward(self, inputs):
        pixel_idx, camrotc2w, campos, near_plane, far_plane = (
            inputs["pixel_idx"].to(torch.int32),
            inputs["camrotc2w"],
            inputs["campos"],
            inputs["near"],
            inputs["far"],
        )

        (
            sample_pidx,
            sample_loc,
            ray_mask_tensor,
            point_xyz_pers_tensor,
            sample_loc_w_tensor,
            sample_ray_dirs_tensor,
            vsize,
        ) = self.get_point_indices(
            inputs,
            camrotc2w,
            campos,
            pixel_idx,
            torch.min(near_plane).cpu().numpy(),
            torch.max(far_plane).cpu().numpy(),
        )
        # (sample_pidx[0, 0] == sample_pidx[0, 1]).sum(-1)
        # tensor([8, 8, 8, 8, 8, 1, 8, 6, 1, 8, 7, 8, 6, 8, 0, 0, 8, 5, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], device='cuda:0')

        sample_pnt_mask = sample_pidx >= 0
        B, R, SR, K = sample_pidx.shape
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()
        sampled_embedding = torch.index_select(
            torch.cat([self.xyz[None, ...], point_xyz_pers_tensor, self.points_embeding[None, ...]], dim=-1),
            1,
            sample_pidx,
        ).view(B, R, SR, K, self.points_embeding.shape[-1] + self.xyz.shape[-1] * 2)

        sampled_color = (
            None
            if self.points_color is None
            else torch.index_select(self.points_color, 1, sample_pidx).view(B, R, SR, K, self.points_color.shape[2])
        )

        sampled_dir = (
            None
            if self.points_dir is None
            else torch.index_select(self.points_dir, 1, sample_pidx).view(B, R, SR, K, self.points_dir.shape[2])
        )

        sampled_Rw2c = (
            self.Rw2c
            if self.Rw2c.dim() == 2
            else torch.index_select(self.Rw2c, 0, sample_pidx).view(B, R, SR, K, self.Rw2c.shape[1], self.Rw2c.shape[2])
        )

        sampled_conf = (
            None
            if self.points_conf is None
            else torch.index_select(self.points_conf, 1, sample_pidx).view(B, R, SR, K, self.points_conf.shape[2])
        )

        return (
            sampled_color,
            sampled_Rw2c,
            sampled_dir,
            sampled_conf,
            sampled_embedding[..., 6:],
            sampled_embedding[..., 3:6],
            sampled_embedding[..., :3],
            sample_pnt_mask,
            sample_loc,
            sample_loc_w_tensor,
            sample_ray_dirs_tensor,
            ray_mask_tensor,
            vsize,
            self.grid_vox_sz,
        )

    def visualize(self):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        o3d.visualization.draw_geometries([pcd])

    def save_points(self, logger, filepath="./neural_points.obj"):
        fw = open(filepath, "w")
        # vertices
        for vert in self.xyz:
            fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

        logger.info("Neural points has saved in " + filepath + ".")
