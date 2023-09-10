# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# obj file dataset
import os
import sys
import sparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image

import torch.distributed as dist


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


def check_path(path):
    if not os.path.exists(path):
        sys.stderr.write("%s does not exist!\n" % (path))
        sys.exit(-1)


def load_krt(path, rate_h=1.0, rate_w=1.0):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = np.array([[float(x) for x in f.readline().split()] for i in range(3)])
            dist = np.array([float(x) for x in f.readline().split()])
            extrin = np.array([[float(x) for x in f.readline().split()] for i in range(3)])
            f.readline()

            camrotc2w = extrin[:3, :3].T
            campos = -camrotc2w @ extrin[:3, 3]

            intrin[0] *= rate_w
            intrin[1] *= rate_h

            cameras[name[:-1]] = {
                "intrin": intrin,
                "dist": dist,
                "extrin": extrin,
                "camrotc2w": camrotc2w,
                "campos": campos,
            }

    return cameras


def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm):
    # rot is c2w
    ## pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs = dirs @ rot[:, :].T  #
    if dir_norm:
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
    # print("dirs", dirs.shape)

    return dirs

# borrow from https://github.com/facebookresearch/multiface/blob/main/dataset.py
class PiCADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        size=1024,
        camset=None,
        valid_prefix=None,
        exclude_prefix=None,
        dir_norm=True,
        is_val=False,
        replay=False,
    ):
        self.rank, self.world_size = config["global_rank"], config["world_size"]
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 2048.0, self.img_w / 1334.0
        self.random_sample, self.random_sample_size = (
            config["rendering.random_sample"],
            config["rendering.random_sample_size"],
        )

        self.err_grid_num_h, self.err_grid_num_w = 8, 8
        self.err_grid_h, self.err_grid_w = (self.img_h // self.err_grid_num_h) + 1, (
            self.img_w // self.err_grid_num_w
        ) + 1

        base_dir, gen_dir = config["data.base_dir"], config["data.gen_dir"]
        self.is_val = is_val
        self.replay = replay

        framelistpath = config["data.train_list"] if not is_val else config["data.test_list"]

        self.uvpath = "{}/unwrapped_uv_1024".format(base_dir)
        self.meshpath = "{}/tracked_mesh".format(base_dir)
        self.photopath = "{}/images".format(base_dir)

        self.depthpath = "{}/depths".format(gen_dir)
        self.posmappath = "{}/posMap".format(gen_dir)
        self.maskpath = "{}/rast_mask".format(gen_dir)

        self.size = size
        self.dir_norm = dir_norm
        self.camera_ids = {}

        krtpath = "{}/KRT".format(base_dir)

        check_path(self.uvpath)
        check_path(self.meshpath)
        check_path(framelistpath)

        framelist = np.genfromtxt(framelistpath, dtype=np.str_)
        if len(framelist.shape) == 1:
            framelist = framelist[None]

        # set cameras
        krt = load_krt(krtpath, self.rate_h, self.rate_w)
        self.krt = krt
        self.cameras = list(krt.keys())
        
        # uncomment this to evaluate one view
        self.cameras = ["400029"] 
        if replay:
            self.cameras = ["400029"]
        for i, k in enumerate(self.cameras):
            self.camera_ids[k] = i

        if camset is not None:
            self.cameras = camset
        self.allcameras = sorted(self.cameras)

        # load train list (but check that images are not dropped!)
        self.framelist = []

        for i, x in enumerate(framelist):
            if i % 1000 == 0:
                print("checking {}".format(i))

            # filter valid prefixes
            if valid_prefix is not None:
                valid = False
                for p in valid_prefix:
                    if x[0].startswith(p):
                        valid = True
                        break
                if not valid:
                    continue

            if exclude_prefix is not None:
                valid = True
                for p in exclude_prefix:
                    if x[0].startswith(p):
                        valid = False
                        break
                if not valid:
                    continue

            # check if has average texture
            avgf = "{}/{}/average/{}.png".format(self.uvpath, x[0], x[1])

            if os.path.isfile(avgf) is not True:
                continue
            # check if has per-view photo
            for i, cam in enumerate(self.cameras):
                f = tuple(x) + (cam,)
                depthpath = "{}/{}/{}/{}.npz".format(self.depthpath, f[0], f[2], f[1])
                maskpath = "{}/{}/{}/{}.npy".format(self.maskpath, f[0], f[2], f[1])
                if os.path.isfile(depthpath) and os.path.isfile(maskpath):
                    self.framelist.append(f)
                    # self.errMap.append(torch.zeros((8, 8), dtype=torch.float32).cuda())
            self.errMap = np.zeros((len(self.framelist), self.err_grid_num_h, self.err_grid_num_w), dtype=np.float32)
            self.counter = np.zeros((len(self.framelist), self.err_grid_num_h, self.err_grid_num_w), dtype=np.int32)

        # compute view directions of each camera
        campos = {}
        for cam in self.cameras:
            extrin = krt[cam]["extrin"]
            campos[cam] = -np.dot(extrin[:3, :3].T, extrin[:3, 3])
        self.campos = campos

        # load mean image and std
        texmean = np.asarray(Image.open("{}/tex_mean.png".format(base_dir)), dtype=np.float32)
        self.texmean = np.copy(np.flip(texmean, 0))
        self.texstd = float(np.genfromtxt("{}/tex_var.txt".format(base_dir)) ** 0.5)
        self.texmin = (np.zeros_like(self.texmean, dtype=np.float32) - self.texmean) / self.texstd
        self.texmax = (np.ones_like(self.texmean, dtype=np.float32) * 255 - self.texmean) / self.texstd

        self.posMean = np.rot90(np.load("{}/posMean.npy".format(gen_dir)), k=1, axes=(1, 2))
        self.posStd = float(np.genfromtxt("{}/posVar.txt".format(gen_dir)) ** 0.5)

    def update_errmap(self, idx, pixel_idx, valid_mask, rgb_loss, gamma=0.1):
        grid_idx = torch.zeros_like(pixel_idx).cuda()
        grid_idx[:, 0] = torch.floor(pixel_idx[:, 0] / self.err_grid_w)
        grid_idx[:, 1] = torch.floor(pixel_idx[:, 1] / self.err_grid_h)

        rgb_loss[~valid_mask] = 0.0

        for i in range(self.err_grid_num_h):
            for j in range(self.err_grid_num_w):
                update_vec = rgb_loss[(grid_idx == torch.tensor([j, i]).cuda()).all(dim=1)]
                if update_vec.shape[0] == 0:
                    continue

                cur_sum = self.errMap[idx][i, j] * self.counter[idx][i, j] + update_vec.sum()
                self.counter[idx][i, j] += update_vec.shape[0]
                self.errMap[idx][i, j] = cur_sum / self.counter[idx][i, j]

        self.errMap[idx] /= self.errMap[idx].sum() + 1e-9

        idx_with_rank = torch.tensor([self.rank, idx.item()]).cuda()
        idx_list = [torch.tensor([0, 0]).cuda() for _ in range(self.world_size)]
        dist.all_gather(idx_list, idx_with_rank)

        # print("idx_list:", idx_list)
        for rank, i in idx_list:
            errMap = torch.from_numpy(self.errMap[i]).cuda()
            counter = torch.from_numpy(self.counter[i]).cuda()
            dist.broadcast(errMap, rank)
            dist.broadcast(counter, rank)
            self.errMap[i] = errMap.cpu().numpy()
            self.counter[i] = counter.cpu().numpy()

    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, idx):
        sentnum, frame, cam = self.framelist[idx]
        cam_id = self.camera_ids[cam]

        transf = np.genfromtxt("{}/{}/{}_transform.txt".format(self.meshpath, sentnum, frame))
        R_f = transf[:3, :3]
        t_f = transf[:3, 3]

        if self.replay:
            coarse_mesh = load_obj("{}/{}/{}.obj".format(self.meshpath, sentnum, frame))
            trans_full = np.concatenate((transf, np.array([[0, 0, 0, 1]])))
            trans_inv = np.linalg.inv(trans_full)[:3]  # 3x4
            verts = np.concatenate(
                (coarse_mesh["verts"], np.ones((coarse_mesh["verts"].shape[0], 1))), axis=-1
            ).transpose(
                1, 0
            )  # 4xN
            coarse_mesh["verts"] = (trans_inv @ verts).transpose(1, 0)  # Nx3

        # posMap
        path = "{}/{}/{}.npy".format(self.posmappath, sentnum, frame)
        posMap = np.rot90(np.load(path), k=1, axes=(0, 1)).transpose((2, 0, 1))
        posMap -= self.posMean
        posMap /= self.posStd

        # average image
        path = "{}/{}/average/{}.png".format(self.uvpath, sentnum, frame)
        avgtex = np.asarray(Image.open(path), dtype=np.float32)
        # avgtex -= self.texmean
        # avgtex /= self.texstd
        avgtex = cv2.resize(avgtex, (self.size, self.size)).transpose((2, 0, 1))  # 0~255

        # load mask
        path = "{}/{}/{}/{}.npy".format(self.maskpath, sentnum, cam, frame)
        mask = (
            cv2.resize(np.load(path).astype(np.float32), (self.img_w, self.img_h), interpolation=cv2.INTER_AREA) > 0
        )  # [img_h, img_w]

        # sample a patch
        self.random_sample = "full" if self.is_val else self.random_sample
        if self.random_sample == "random":
            px = np.random.randint(0, self.img_w, size=(self.random_sample_size, self.random_sample_size)).astype(
                np.float32
            )
            py = np.random.randint(0, self.img_h, size=(self.random_sample_size, self.random_sample_size)).astype(
                np.float32
            )
        elif self.random_sample == "patch":
            coords = np.meshgrid(range(self.img_w), range(self.img_h))
            coords = np.stack(coords, axis=-1)  # [img_h, img_w, 2]
            mask_coords = coords[mask]  # [N, 2]
            radius = self.random_sample_size // 2
            valid = (
                (mask_coords[:, 0] >= radius)
                & (mask_coords[:, 1] >= radius)
                & (mask_coords[:, 0] <= (self.img_w - radius))
                & (mask_coords[:, 1] <= (self.img_h - radius))
            )
            valid_coords = mask_coords[valid]

            i = np.random.randint(valid_coords.shape[0])
            center = valid_coords[i]

            px, py = np.meshgrid(
                np.arange(center[0] - radius, center[0] + radius).astype(np.float32),
                np.arange(center[1] - radius, center[1] + radius).astype(np.float32),
            )
        elif self.random_sample == "grid":
            grid_w = int(4 * self.rate_w)
            grid_h = int(6 * self.rate_h)

            w = np.random.randint(grid_w)
            h = np.random.randint(grid_h)
            px, py = np.meshgrid(
                np.arange(w, self.img_w, grid_w).astype(np.float32), np.arange(h, self.img_h, grid_h).astype(np.float32)
            )
        elif self.random_sample == "err":
            total_sample = self.random_sample_size**2
            numMap = np.round(self.errMap[idx] * total_sample).astype(np.int32)  # 8x8, sum=1
            px, py = None, None
            if numMap.max() == 0:
                grid_w = int(4 * self.rate_w)
                grid_h = int(6 * self.rate_h)

                w = np.random.randint(grid_w)
                h = np.random.randint(grid_h)
                px, py = np.meshgrid(
                    np.arange(w, self.img_w, grid_w).astype(np.float32),
                    np.arange(h, self.img_h, grid_h).astype(np.float32),
                )
            else:
                max_i, max_j = numMap.argmax() // 8, numMap.argmax() % 8
                numMap[max_i, max_j] += total_sample - numMap.sum()

                for i in range(8):
                    for j in range(8):
                        patch_px = np.random.randint(
                            self.err_grid_w * j, min(self.err_grid_w * (j + 1), self.img_w), size=numMap[i, j]
                        )
                        patch_py = np.random.randint(
                            self.err_grid_h * i, min(self.err_grid_h * (i + 1), self.img_h), size=numMap[i, j]
                        )

                        if px is None:
                            px = patch_px
                            py = patch_py
                        else:
                            px = np.concatenate([px, patch_px], axis=-1)
                            py = np.concatenate([py, patch_py], axis=-1)
        else:
            px, py = np.meshgrid(np.arange(self.img_w).astype(np.float32), np.arange(self.img_h).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32).reshape(-1, 2)  # HW x 2

        # depth map, [img_h, img_w, 1]
        path = "{}/{}/{}/{}.npz".format(self.depthpath, sentnum, cam, frame)
        depth_map = sparse.load_npz(path).todense()
        depth_map = cv2.resize(depth_map, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        if self.is_val:
            depth_map = depth_map * 1000
        else:
            depth_map = (depth_map * 1000)[py.astype(np.int32), px.astype(np.int32)].reshape(-1)  # [p*p]

        # image
        path = "{}/{}/{}/{}.png".format(self.photopath, sentnum, cam, frame)
        photo = np.asarray(Image.open(path), dtype=np.float32)
        photo = cv2.resize(photo, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        if self.is_val:
            photo = photo / 255.0
        else:
            photo = (photo / 255.0)[py.astype(np.int32), px.astype(np.int32)].reshape(-1, 3)  # [p*p, 3]

        # view direction
        campos = np.dot(R_f.T, self.campos[cam] - t_f).astype(np.float32)
        view = campos / np.linalg.norm(campos)
        view_tile = np.tile(view, (8, 8, 1)).transpose((2, 0, 1))

        extrin, intrin = self.krt[cam]["extrin"], self.krt[cam]["intrin"]
        R_C = extrin[:3, :3]
        t_C = extrin[:3, 3]
        camrot = np.dot(R_C, R_f).astype(np.float32)
        camt = np.dot(R_C, t_f) + t_C
        camt = camt.astype(np.float32)

        M = intrin @ np.hstack((camrot, camt[None].T))

        # ray direction
        camrotc2w, campos = self.krt[cam]["camrotc2w"], self.krt[cam]["campos"]
        raydir = get_dtu_raydir(pixelcoords, intrin, camrotc2w, self.dir_norm)
        if self.is_val:
            raydir = raydir.reshape(self.img_h, self.img_w, 3)
            pixelcoords = pixelcoords.reshape(self.img_h, self.img_w, 2)

        res = {
            "idx": idx,
            "cam_idx": cam,
            "frame": frame,
            "exp": sentnum,
            "cam": cam_id,
            "M": M.astype(np.float32),
            "posMap": posMap.astype(np.float32),
            "depth_map": depth_map.astype(np.float32),
            "avg_tex": avgtex,
            "mask": mask,
            "view": view_tile,
            "transf": transf.astype(np.float32),
            "photo": photo,
            # neural points
            "pixel_idx": pixelcoords.astype(np.int32),
            "raydir": raydir.astype(np.float32),
            "camrotc2w": camrotc2w.astype(np.float32),
            "campos": campos.astype(np.float32),
            "intrinsic": intrin.astype(np.float32),
            "camrotf2w": R_f.astype(np.float32),
            "camtf2w": t_f.astype(np.float32),
        }

        if self.replay:
            res["coarse_verts"] = coarse_mesh["verts"]
            res["coarse_tris"] = coarse_mesh["vert_ids"]
            res["P_f2c"] = np.hstack((camrot, camt[None].T))

        return res
