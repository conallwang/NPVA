import argparse
import torch
import torch.nn.functional as F
import yaml
import os, sys, shutil
import cv2
import math

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from dataset.pica_dataset import get_dtu_raydir, load_krt, PiCADataset
from utils import seed_everything, visDepthMap, visPositionMap, write_obj
from trainer import Trainer
from mesh_viewer import MeshViewer

from PIL import Image
from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser("REPLAY VIDEOS")
parser.add_argument("--img_list", type=str, required=True, help="image path of an expression sequence")
parser.add_argument("--checkpoint", type=str, required=True, help="path of model checkpoint")
parser.add_argument("--mode", type=str, default="cat", help="mix mode of image and geometry")
parser.add_argument("--only_video", action="store_true", help="only render videos from saved images")
parser.add_argument("--only_mesh", action="store_true", help="only render mesh")
parser.add_argument("--total_frames", type=int, default=-1)
parser.add_argument("--fix_view", action="store_true", help="if fixing view")
parser.add_argument("--fix_view_id", type=int, default=-1, help="when fixing view, which view is used")
parser.add_argument("--fix_expr", action="store_true", help="if fixing expression")
parser.add_argument("--fix_expr_id", type=int, default=0, help="when fixing expression, which z_code is used")
parser.add_argument("--prmesh", action="store_true", help="if using pyrender to render coarse mesh")
parser.add_argument("--name", type=str, default="", help="identify different videos")

args = parser.parse_args()

# make sure params.yaml in the same directory with checkpoint
dir_name = os.path.dirname(args.checkpoint)
config_path = os.path.join(dir_name, "params.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["data.test_list"] = args.img_list
chunk = 50000
num_between = 0

config["global_rank"] = 0
config["world_size"] = 1
seed_everything(42)

krtpath = "{}/KRT".format(config["data.base_dir"])
img_h, img_w = config["data.img_h"], config["data.img_w"]
rate_h, rate_w = img_h / 2048.0, img_w / 1334.0
krt = load_krt(krtpath, rate_h, rate_w)


cam_ids = ["400029", "400042"]
s_view, e_view = krt[cam_ids[0]]["extrin"], krt[cam_ids[1]]["extrin"]
intrin = krt[cam_ids[0]]["intrin"]

tris = np.load('./assets/data/tris.npy') - 1

def directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError as e:
            print(path + " exists. (multiprocess conflict)")


def depth_map2normals(depth_map):
    """convert a depth map to normal map

    Args:
        depth_map: HxW
    """
    zy, zx = np.gradient(depth_map)
    normals = np.dstack((-zx, -zy, np.ones_like(depth_map)))

    n = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= n

    normals += 1
    normals /= 2

    return normals


def gammaCorrect(img, dim: int = -1):
    if dim == -1:
        dim = len(img.shape) - 1
    assert img.shape[dim] == 3
    gamma, black, color_scale = 2.0, 3.0 / 255.0, [1.4, 1.1, 1.6]

    img = img[..., [2, 1, 0]]
    if dim == -1:
        dim = len(img.shape) - 1

    scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
    img = img * scale / 1.1
    return np.clip(
        (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0,
        0,
        2,
    )


def lerp(start, end, t):
    if start is None or end is None:
        return None
    return (1 - t) * start + t * end


def slerp(start, end, t):
    rot_mats = np.stack([start, end], axis=0)  # [2, 3, 3]
    key_rots = R.from_matrix(rot_mats)
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(t).as_matrix()  # [3, 3]
    return rot


def interp_extrin(start_extrin, end_extrin, t):
    """Interpolate extrin matrix

    Args:
        start_extrin (float): 4x4 extrin matrix, numpy
        end_extrin (float): 4x4 extrin matrix, numpy
        t (float):
    """
    s_rot, s_trans = start_extrin[:3, :3], start_extrin[:3, 3:]
    e_rot, e_trans = end_extrin[:3, :3], end_extrin[:3, 3:]
    rot = slerp(s_rot, e_rot, t)
    trans = (1 - t) * s_trans + t * e_trans  # [3, 1]
    return np.concatenate([rot, trans], axis=-1)

# from up to down
# def create_spheric_poses(n_steps=120):
#     center = np.array([7.0, 10.0, 1063.0]).astype(np.float32)
#     r = 1200
#     up = np.array([0.0, -1.0, 0.0], dtype=center.dtype)

#     all_c2w = []
#     for theta in np.linspace(-2 * math.pi / 3, -math.pi / 3, n_steps):
#         diff = np.stack([0, r * np.cos(theta), r * np.sin(theta)])
#         cam_pos = center + diff
#         l = -diff / np.linalg.norm(diff)
#         s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
#         u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
#         c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
#         all_c2w.append(c2w)

#     all_c2w = np.stack(all_c2w, axis=0)

#     return all_c2w


# from left to right
# def create_spheric_poses(n_steps=120):
#     center = np.array([7.0, 10.0, 1063.0]).astype(np.float32)
#     r = 1200
#     up = np.array([0.0, -1.0, 0.0], dtype=center.dtype)

#     all_c2w = []
#     for theta in np.linspace(-2 * math.pi / 3, -math.pi / 3, n_steps):
#         diff = np.stack([r * np.cos(theta), 0, r * np.sin(theta)])
#         cam_pos = center + diff
#         l = -diff / np.linalg.norm(diff)
#         s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
#         u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
#         c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
#         all_c2w.append(c2w)

#     all_c2w = np.stack(all_c2w, axis=0)

#     return all_c2w

# 0105
def create_spheric_poses(n_steps=120):
    center = np.array([7.0, 200.0, 1063.0]).astype(np.float32)
    r = 1200
    up = np.array([0.0, -1.0, 0.0], dtype=center.dtype)

    all_c2w = []
    for theta in np.linspace(-2 * math.pi / 3, -math.pi / 3, n_steps):
        diff = np.stack([r * np.cos(theta), 0, r * np.sin(theta)])
        cam_pos = center + diff
        l = -diff / np.linalg.norm(diff)
        s = np.cross(l, up) / np.linalg.norm(np.cross(l, up))
        u = np.cross(s, l) / np.linalg.norm(np.cross(s, l))
        c2w = np.concatenate([np.stack([s, -u, l], axis=1), cam_pos[:, None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = np.stack(all_c2w, axis=0)

    return all_c2w


def render(trainer, data, z_code):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = torch.from_numpy(v[None])

    trainer.set_data(data, is_val=True)

    dec_res = trainer.net_decode(z_code)
    # nvdiffrast: rasterization
    posMap = dec_res["posMap"]
    rast_res = trainer.diff_rasterization(posMap.reshape(posMap.shape[0], 3, -1).permute((0, 2, 1)))

    # neural points ray marching
    render_res = trainer.render_all_pixels(rast_res, dec_res, chunk=chunk)
    return render_res, rast_res["valid_pixels"], posMap


def make_videos(img_list, geo_list, mesh_list, savepath, mode="cat"):
    """make videos from image list and geometry list

    Args:
        img_list (list): image list
        geo_list (list): geometry list (normals list)
        savepath (str): save path
        mode (str, optional): mode to show image and geometry. Defaults to 'cat'. SUPPORT: 'cat', 'mix'
    """
    assert len(img_list) == len(geo_list)
    img_h, img_w, _ = img_list[0].shape

    n_frames = len(img_list)
    if mode == "cat":
        v_h, v_w = img_h, 3 * img_w
    elif mode == "mix":
        v_h, v_w = img_h, img_w
        ws = np.linspace(0, v_w, n_frames)

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"HFYU")
    video = cv2.VideoWriter(savepath, fourcc, fps=30, frameSize=(v_w, v_h))

    for i in range(n_frames):
        if mode == "cat":
            frame = np.concatenate([mesh_list[i], img_list[i], geo_list[i]], axis=1)  # [img_h, 2 * img_w, 3]
        elif mode == "mix":
            w = int(np.ceil(ws[i]))
            mask = np.zeros((img_h, img_w, 1)).astype(np.bool_)
            mask[:, :w] = True
            frame = geo_list[i] * mask + img_list[i] * (~mask)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    
def make_mesh_video(mesh_list, savepath):
    img_h, img_w, _ = mesh_list[0].shape
    n_frames = len(mesh_list)

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"HFYU")
    video = cv2.VideoWriter(savepath, fourcc, fps=30, frameSize=(img_w, img_h))

    for i in range(n_frames):
        frame = mesh_list[i]
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    

def get_data(views, view_idx, data_cur, z_cur, t=0, z_next=None, data_next=None):
    if z_next is None:
        z_code = z_cur

        R_f = data_cur["camrotf2w"]
        t_f = data_cur["camtf2w"]
    else:
        z_code = lerp(z_cur, z_next, t)

        R_f = slerp(data_cur["camrotf2w"], data_next["camrotf2w"], t)
        t_f = lerp(data_cur["camtf2w"], data_next["camtf2w"], t)

    view_idx = args.fix_view_id if args.fix_view else view_idx
    if view_idx == -1:
        R_C = s_view[:3, :3]
        t_C = s_view[:3, 3]
        
        camrotc2w = R_C.T
        campos = -camrotc2w @ t_C
    else:
        view = np.concatenate([s_view, np.array([[0, 0, 0, 1]])], axis=0) @ np.concatenate([views[view_idx], np.array([[0, 0, 0, 1]])], axis=0)
        
        camrotc2w = view[:3, :3]
        campos = view[:3, 3]

        R_C = camrotc2w.T
        t_C = -R_C @ campos

    camrot = np.dot(R_C, R_f).astype(np.float32)
    camt = np.dot(R_C, t_f) + t_C
    camt = camt.astype(np.float32)

    M = intrin @ np.hstack((camrot, camt[None].T))

    pixelcoords = data_cur["pixel_idx"]  # [H, W, 2]
    raydir = get_dtu_raydir(pixelcoords, intrin, camrotc2w, dir_norm=True)

    data = deepcopy(data_cur)
    data["M"] = M.astype(np.float32)
    data["raydir"] = raydir.astype(np.float32)
    data["camrotc2w"] = camrotc2w.astype(np.float32)
    data["campos"] = campos.astype(np.float32)
    data["intrinsic"] = intrin.astype(np.float32)
    data["camrotf2w"] = R_f.astype(np.float32)
    data["camtf2w"] = t_f.astype(np.float32)
    data["P_f2c"] = np.hstack((camrot, camt[None].T))

    return data, z_code


def camera_intrinsic_fov(intrinsic):
    # 计算FOV
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    # Go
    fov_x = 2 * np.arctan2(img_w, 2 * fx)
    fov_y = 2 * np.arctan2(img_h, 2 * fy)
    return fov_x, fov_y


def only_mesh(logger):
    global s_view

    dataset = PiCADataset(config, is_val=True, replay=True)
    n_frames = len(dataset)

    meshviewer = MeshViewer(img_h, img_w)
    
    seq_name = dataset[0]["exp"]
    seq_dir = os.path.join(args.output_dir, seq_name + "_" + args.name)
    mesh_dir = os.path.join(seq_dir, "mesh")
    if os.path.exists(seq_dir):
        if args.only_video and os.path.exists(mesh_dir):
            meshpath_list = [os.path.join(mesh_dir, img) for img in os.listdir(mesh_dir) if img.endswith(".png")]
            meshpath_list.sort()
            
            mesh_list = [cv2.imread(img) for img in meshpath_list]

            logger.info("start rendering video ... [MODE] {}".format(args.mode))
            savepath = os.path.join(seq_dir, "video.avi")
            make_mesh_video(mesh_list, savepath)
            return
        shutil.rmtree(seq_dir)
    directory(seq_dir)
    directory(mesh_dir)
    
    total_frames = (num_between + 1) * (n_frames - 1) + 1
    views = create_spheric_poses(n_steps=total_frames)
    view_idx = 0

    mesh_list = []

    if args.fix_expr:
        data_cur = dataset[args.fix_expr_id]
        z_fix = None

    bar = tqdm(range(total_frames))
    iters = total_frames if args.fix_expr else n_frames
    for i in range(iters):
        if args.fix_expr:
            data_cur = dataset[args.fix_expr_id]
            z_cur = None
        else:
            data_cur = dataset[i]
            z_cur = None

        if args.fix_expr or i == (n_frames - 1):
            data, z_code = get_data(views, view_idx, data_cur, z_cur)

            # save coarse mesh
            if args.prmesh:
                meshviewer.create_mesh(data_cur["coarse_verts"], data_cur["coarse_tris"])
                color, depth = meshviewer.render(data["P_f2c"], data_cur["intrinsic"])
            else:
                T_verts = torch.from_numpy(data_cur["coarse_verts"][None]).float().cuda()
                T_tris = torch.from_numpy(data_cur["coarse_tris"][None]).int().cuda()
                T_M = torch.from_numpy(data["M"][None]).float().cuda()
                rast_out, rast_attr = trainer.renderer.render(T_M, T_verts, T_tris)
                mesh_valid = (rast_out[0, :trainer.img_h, :trainer.img_w, 3] > 0).detach().cpu().numpy()
                rast_depth = rast_attr[0, :trainer.img_h, :trainer.img_w, 0].detach().cpu().numpy()  # [H, W]
                color = (depth_map2normals(rast_depth)[..., ::-1] * 255).astype(np.uint8) * mesh_valid[..., None]
            mesh_list.append(color)
            savepath = os.path.join(mesh_dir, f"{view_idx:03}.png")
            cv2.imwrite(savepath, color)

            view_idx += 1
            bar.update()
            continue

        data_next = dataset[i + 1]
        z_next = None

        ts = np.linspace(0, 1, num_between + 2)[:-1]
        for t in ts:
            data, z_code = get_data(views, view_idx, data_cur, z_cur, t, z_next, data_next)

            # save coarse mesh
            if args.prmesh:
                meshviewer.create_mesh(data_cur["coarse_verts"], data_cur["coarse_tris"])
                color, depth = meshviewer.render(data["P_f2c"], data_cur["intrinsic"])
            else:
                T_verts = torch.from_numpy(data_cur["coarse_verts"][None]).float().cuda()
                T_tris = torch.from_numpy(data_cur["coarse_tris"][None]).int().cuda()
                T_M = torch.from_numpy(data["M"][None]).float().cuda()
                rast_out, rast_attr = trainer.renderer.render(T_M, T_verts, T_tris)
                mesh_valid = (rast_out[0, :trainer.img_h, :trainer.img_w, 3] > 0).detach().cpu().numpy()
                rast_depth = rast_attr[0, :trainer.img_h, :trainer.img_w, 0].detach().cpu().numpy()  # [H, W]
                color = (depth_map2normals(rast_depth)[..., ::-1] * 255).astype(np.uint8) * mesh_valid[..., None]
            mesh_list.append(color)
            savepath = os.path.join(mesh_dir, f"{view_idx:03}.png")
            cv2.imwrite(savepath, color)

            view_idx += 1
            bar.update()

    # Make Videos
    logger.info("start rendering video ... [MODE] {}".format(args.mode))
    savepath = os.path.join(seq_dir, "video.avi")
    make_mesh_video(mesh_list, savepath)
    

def eval_image(trainer, logger):
    global s_view

    dataset = PiCADataset(config, is_val=True, replay=True)
    n_frames = len(dataset)

    meshviewer = MeshViewer(img_h, img_w)

    seq_name = dataset[0]["exp"]
    seq_dir = os.path.join(args.output_dir, seq_name + "_" + args.name)
    img_dir = os.path.join(seq_dir, "images")
    geo_dir = os.path.join(seq_dir, "geometrys")
    mesh_dir = os.path.join(seq_dir, "mesh")
    # surface_dir = os.path.join(seq_dir, "surface")
    if os.path.exists(seq_dir):
        if args.only_video and os.path.exists(img_dir) and os.path.exists(geo_dir) and os.path.exists(mesh_dir):
            # imgpath_list = [img for img in os.listdir(img_dir) if img.endswith('.png')]
            # geopath_list = [img for img in os.listdir(geo_dir) if img.endswith('.png')]
            imgpath_list = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".png")]
            geopath_list = [os.path.join(geo_dir, img) for img in os.listdir(geo_dir) if img.endswith(".png")]
            meshpath_list = [os.path.join(mesh_dir, img) for img in os.listdir(mesh_dir) if img.endswith(".png")]
            imgpath_list.sort()
            geopath_list.sort()
            meshpath_list.sort()

            img_list = [cv2.imread(img) for img in imgpath_list]
            geo_list = [cv2.imread(img) for img in geopath_list]
            mesh_list = [cv2.imread(img) for img in meshpath_list]

            logger.info("start rendering video ... [MODE] {}".format(args.mode))
            savepath = os.path.join(seq_dir, "video.avi")
            make_videos(img_list, geo_list, mesh_list, savepath, mode=args.mode)
            return
        shutil.rmtree(seq_dir)
    directory(seq_dir)
    directory(img_dir)
    directory(geo_dir)
    directory(mesh_dir)
    # directory(surface_dir)

    if args.total_frames == -1:
        total_frames = (num_between + 1) * (n_frames - 1) + 1
    else:
        total_frames = args.total_frames
    views = create_spheric_poses(n_steps=total_frames)
    view_idx = 0

    img_list = []
    geo_list = []
    mesh_list = []

    if args.fix_expr:
        data_cur = dataset[args.fix_expr_id]
        avgTex, posMap = (
            torch.from_numpy(data_cur["avg_tex"][None]).cuda(),
            torch.from_numpy(data_cur["posMap"][None]).cuda(),
        )
        z_fix = trainer.encode(avgTex, posMap, is_val=True)["z_code"]

    bar = tqdm(range(total_frames))
    iters = total_frames if args.fix_expr else n_frames
    for i in range(iters):
        if args.fix_expr:
            data_cur = dataset[args.fix_expr_id]
            z_cur = z_fix
        else:
            data_cur = dataset[i]
            avgTex, posMap = (
                torch.from_numpy(data_cur["avg_tex"][None]).cuda(),
                torch.from_numpy(data_cur["posMap"][None]).cuda(),
            )
            z_cur = trainer.encode(avgTex, posMap, is_val=True)["z_code"]

        if args.fix_expr or i == (n_frames - 1):
            data, z_code = get_data(views, view_idx, data_cur, z_cur)

            render_res, valid, posMap = render(trainer, data, z_code)
            valid = valid[0].detach().cpu().numpy()

            # save coarse mesh
            if args.prmesh:
                meshviewer.create_mesh(data_cur["coarse_verts"], data_cur["coarse_tris"])
                color, depth = meshviewer.render(data["P_f2c"][0], data_cur["intrinsic"])
            else:
                T_verts = torch.from_numpy(data_cur["coarse_verts"][None]).float().cuda()
                T_tris = torch.from_numpy(data_cur["coarse_tris"][None]).int().cuda()
                _, rast_attr = trainer.renderer.render(data["M"].cuda(), T_verts, T_tris)
                rast_depth = rast_attr[0, : trainer.img_h, : trainer.img_w, 0].detach().cpu().numpy()  # [H, W]
                color = (depth_map2normals(rast_depth)[..., ::-1] * 255).astype(np.uint8)
            mesh_list.append(color)
            savepath = os.path.join(mesh_dir, f"{view_idx:03}.png")
            cv2.imwrite(savepath, color)

            # save images
            savepath = os.path.join(img_dir, f"{view_idx:03}.png")
            photo = render_res["photo"][0].detach().cpu().numpy()
            color_corr = lambda x: np.clip(255 * gammaCorrect(x, dim=-1), 0, 255)
            img = color_corr(photo).astype(np.uint8)
            img_list.append(img)
            cv2.imwrite(savepath, img)

            # save geometry
            savepath = os.path.join(geo_dir, f"{view_idx:03}.png".format(i))
            depth = render_res["depth"][0].detach().cpu().numpy()
            normals = (depth_map2normals(depth)[..., ::-1] * 255).astype(np.uint8)
            normals[~valid] = 0.0
            geo_list.append(normals)
            cv2.imwrite(savepath, normals)

            view_idx += 1
            bar.update()
            continue

        data_next = dataset[i + 1]
        avgTex, posMap = (
            torch.from_numpy(data_next["avg_tex"][None]).cuda(),
            torch.from_numpy(data_next["posMap"][None]).cuda(),
        )
        z_next = trainer.encode(avgTex, posMap, is_val=True)["z_code"]

        ts = np.linspace(0, 1, num_between + 2)[:-1]
        for t in ts:
            data, z_code = get_data(views, view_idx, data_cur, z_cur, t, z_next, data_next)

            render_res, valid, _ = render(trainer, data, z_code)
            valid = valid[0].detach().cpu().numpy()
            
            # save coarse mesh
            if args.prmesh:
                meshviewer.create_mesh(data_cur["coarse_verts"], data_cur["coarse_tris"])
                color, depth = meshviewer.render(data["P_f2c"][0], data_cur["intrinsic"])
            else:
                T_verts = torch.from_numpy(data_cur["coarse_verts"][None]).float().cuda()
                T_tris = torch.from_numpy(data_cur["coarse_tris"][None]).int().cuda()
                _, rast_attr = trainer.renderer.render(data["M"].cuda(), T_verts, T_tris)
                rast_depth = rast_attr[0, : trainer.img_h, : trainer.img_w, 0].detach().cpu().numpy()  # [H, W]
                color = (depth_map2normals(rast_depth)[..., ::-1] * 255).astype(np.uint8)
            mesh_list.append(color)
            savepath = os.path.join(mesh_dir, f"{view_idx:03}.png")
            cv2.imwrite(savepath, color)

            # save images
            savepath = os.path.join(img_dir, f"{view_idx:03}.png")
            photo = render_res["photo"][0].detach().cpu().numpy()
            color_corr = lambda x: np.clip(255 * gammaCorrect(x, dim=-1), 0, 255)
            img = color_corr(photo).astype(np.uint8)
            img_list.append(img)
            cv2.imwrite(savepath, img)

            # save geometry
            savepath = os.path.join(geo_dir, f"{view_idx:03}.png".format(i))
            depth = render_res["depth"][0].detach().cpu().numpy()
            normals = (depth_map2normals(depth)[..., ::-1] * 255).astype(np.uint8)
            normals[~valid] = 0.0
            geo_list.append(normals)
            cv2.imwrite(savepath, normals)

            view_idx += 1
            bar.update()

    # Make Videos
    logger.info("start rendering video ... [MODE] {}".format(args.mode))
    savepath = os.path.join(seq_dir, "video.avi")
    make_videos(img_list, geo_list, mesh_list, savepath, mode=args.mode)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = False

    args.output_dir = os.path.join(dir_name, "videos")
    directory(args.output_dir)

    # Config logging and tb writer
    logger = None
    import logging

    # logging to file and stdout
    # config["log_file"] = os.path.join(dir_name, 'test_image.log')
    logger = logging.getLogger("NPA")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s %(filename)s] %(message)s")
    stream_handler.setFormatter(formatter)
    # file_handler = logging.FileHandler(config["log_file"])
    # file_handler.setFormatter(formatter)
    # logger.handlers = [file_handler, stream_handler]
    logger.handlers = [stream_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    config["training.pretrained_checkpoint_path"] = args.checkpoint
    config["logger"] = logger
    config["global_rank"] = 0
    config["local_rank"] = 0

    logger.info("Config: {}".format(config))

    trainer = Trainer(config, logger, is_val=True)
    trainer.set_eval()

    with torch.no_grad():
        if args.only_mesh:
            only_mesh(logger)
        else:
            eval_image(trainer, logger)
