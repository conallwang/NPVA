import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from scipy.spatial import Delaunay
from tqdm import tqdm
from matplotlib import pyplot as plt

# DEBUG
# from dataset.pica_dataset import load_obj

root = "/path/to/Neural_Point-based_Avatars"


def upsample(posMap):
    """upsample 2x posMap

    Args:
        posMap: [B, 3, H, W], tensor
    """
    if len(posMap.shape) == 3:
        posMap = posMap.unsqueeze(0).float()

    B, _, H, W = posMap.shape
    res = torch.zeros((B, 3, H * 2 + 1, W * 2 + 1)).to(posMap.device)
    hh, ww = torch.meshgrid(torch.arange(0, H * 2, 2), torch.arange(0, W * 2, 2))
    res[..., hh, ww] = posMap
    res[..., H * 2, :] = res[..., H * 2 - 2, :]
    res[..., W * 2] = res[..., W * 2 - 2]

    res[..., hh, ww + 1] = (res[..., hh, ww + 2] + res[..., hh, ww]) / 2
    res[..., hh + 1, ww] = (res[..., hh, ww] + res[..., hh + 2, ww]) / 2
    res[..., hh + 1, ww + 1] = (res[..., hh, ww] + res[..., hh + 2, ww + 2]) / 2

    return res[..., : H * 2, : W * 2]


def downsample(posMap, method="fix", rate=4):
    """downsample Nx posMap

    Args:
        posMap: [B, C, H, W], tensor
        method (str, optional): Defaults to 'fix', can choose 'mean'.
    """
    B, _, H, W = posMap.shape
    if method == "fix":
        h = torch.tensor(range(0, H, rate)).cuda()
        w = torch.tensor(range(0, W, rate)).cuda()
        hh, ww = torch.meshgrid(h, w)
        res = posMap[:, :, hh, ww]
    elif method == "mean":
        raise "Not Implemented Error."
    else:
        raise "Not Implemented Error."

    return res


def hook(tensor):
    tensor.register_hook(
        lambda x: print("Mean Grad: " + str(x.abs().mean().data) + "; Max Grad: " + str(x.abs().max().data))
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timing(logger, last_t, label=""):
    logger.info(label + " : {}ms".format((time.time() - last_t) * 1000))
    return time.time()


def update_lambda(lambda_start, lambda_slope, lambda_end, global_step, interval):
    res = lambda_start
    if lambda_slope > 0:
        res = min(lambda_end, global_step // interval * lambda_slope + lambda_start)
    elif lambda_slope < 0:
        res = max(lambda_end, global_step // interval * lambda_slope + lambda_start)
    return res


def restore_model(model_path, models, optimizer, train_loader, logger):
    """Restore checkpoint

    Args:
        model_path (str): checkpoint path
        models (dict): model dict
        optimizer (optimizer): torch optimizer
        logger (logger): logger
    """
    if model_path is None:
        if logger:
            logger.info("Not using pre-trained model...")
        return 1

    assert os.path.exists(model_path), "Model %s does not exist!"
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())

    for key, model in models.items():
        _state_dict = {
            k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict[key].items()
        }
        # Check if there is key mismatch:
        missing_in_model = set(_state_dict.keys()) - set(model.state_dict().keys())
        missing_in_ckp = set(model.state_dict().keys()) - set(_state_dict.keys())

        if logger:
            logger.info("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
            logger.info("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))

        model.load_state_dict(_state_dict, strict=False)

    # load optimizer
    optimizer.load_state_dict(state_dict["optimizer"])

    # load errMap
    if train_loader is not None and "lossmap" in state_dict:
        train_loader.dataset.errMap = state_dict["lossmap"]

    current_epoch = state_dict["epoch"] if "epoch" in state_dict else 1
    global_step = state_dict["global_step"] if "global_step" in state_dict else 0

    return current_epoch, global_step


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # If you want to reproduce fully, this item should be set False
    # But that will reduce the performance
    torch.backends.cudnn.benchmark = True


def upTo8Multiples(num):
    """up to an 8 multiple

    Args:
        num (int): any number
    """
    res = num
    if num % 8:
        res = 8 * (num // 8 + 1)
    return res


def posmap_sample(posMap, uvs):
    """Sample posMap with fixed mesh topology

    Args:
        posMap (float): [B, 3, 256, 256]
        uvs (float): [B, N_uv, 2], 0~1

    Returns:
        samples: [B, N_uv, 3]
    """
    uvs = 2 * (uvs[:, :, None, :] - 0.5)  # -1~1
    samples = F.grid_sample(posMap, uvs, mode="bilinear", align_corners=True)[..., 0]  # [B, 3, N_uv]

    return samples.permute(0, 2, 1)


def truncated_l1_loss(pred, gt, reduction="mean", beta=1.0, rate=1.0):
    delta_map = torch.abs(pred - gt)
    mask = delta_map > beta
    if mask.sum() == 0:
        return torch.tensor(0.0).cuda()

    loss = rate * (delta_map[mask] - beta)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise NotImplementedError

    return loss


def gradient_smooth(posMap, method="smooth_l1", beta=8):
    """Compute the gradient smooth item

    Args:
        posMap (float): [B, 3, 256, 256]

    Returns:
        _type_: _description_
    """
    if method == "l1":
        diff_x = torch.abs(posMap[:, :, 1:, :] - posMap[:, :, :-1, :]).mean()
        diff_y = torch.abs(posMap[..., 1:] - posMap[..., :-1]).mean()
    elif method == "l2":
        diff_x = F.mse_loss(posMap[:, :, 1:, :], posMap[:, :, :-1, :], reduction="mean")
        diff_y = F.mse_loss(posMap[..., 1:], posMap[..., :-1], reduction="mean")
    elif method == "smooth_l1":
        diff_x = F.smooth_l1_loss(posMap[:, :, 1:, :], posMap[:, :, :-1, :], reduction="mean", beta=beta)
        diff_y = F.smooth_l1_loss(posMap[..., 1:], posMap[..., :-1], reduction="mean", beta=beta)
    elif method == "truncated_l1":
        diff_x = truncated_l1_loss(posMap[:, :, 1:, :], posMap[:, :, :-1, :], reduction="mean", beta=beta)
        diff_y = truncated_l1_loss(posMap[..., 1:], posMap[..., :-1], reduction="mean", beta=beta)
    else:
        raise NotImplementedError

    return diff_x + diff_y


# Too much memory !
# def batchBarycentric2d(tri, points):
#     """
#     Args:
#         tri Nx3x2: vertices of triangles
#         points Mx2: query points
#     RET:
#         barycentric coordinates     NxMx3
#     """
#     eps = 1e-9
#     M = points.shape[0]

#     PA = tri[:, 0:1].repeat(M, axis=1) - points  # NxMx2
#     PB = tri[:, 1:2].repeat(M, axis=1) - points
#     PC = tri[:, 2:3].repeat(M, axis=1) - points
#     AB = tri[:, 1] - tri[:, 0]  # Nx2
#     AC = tri[:, 2] - tri[:, 0]

#     PAB = np.abs(np.cross(PA, PB))      # NxM
#     PBC = np.abs(np.cross(PB, PC))
#     PAC = np.abs(np.cross(PA, PC))
#     ABC = np.abs(np.cross(AB, AC))[:, None].repeat(M, axis=1)      # N

#     u = PBC / (ABC + eps)       # NxM
#     v = PAC / (ABC + eps)
#     w = PAB / (ABC + eps)

#     return np.stack([u, v, w], axis=-1)


def batchBarycentric2d(tri, P):
    """
    Args:
        tri Nx3x2: vertices of triangles
        points 2: query points
    RET:
        barycentric coordinates     Nx3
    """
    eps = 1e-9

    PA = tri[:, 0] - P  # Nx2
    PB = tri[:, 1] - P
    PC = tri[:, 2] - P
    AB = tri[:, 1] - tri[:, 0]
    AC = tri[:, 2] - tri[:, 0]

    PAB = np.abs(np.cross(PA, PB))  # N
    PBC = np.abs(np.cross(PB, PC))
    PAC = np.abs(np.cross(PA, PC))
    ABC = np.abs(np.cross(AB, AC))

    u = PBC / (ABC + eps)  # N
    v = PAC / (ABC + eps)
    w = PAB / (ABC + eps)

    return np.stack([u, v, w], axis=-1)


def barycentric2d(tri, P):
    """
    Args:
        tri 3x2: three vertices of the triangle
        P 2: query point
    RET:
        barycentric coordinates
    """
    PA = tri[0] - P
    PB = tri[1] - P
    PC = tri[2] - P
    AB = tri[1] - tri[0]
    AC = tri[2] - tri[0]

    PAB = np.cross(PA, PB)
    PBC = np.cross(PB, PC)
    PAC = np.cross(PA, PC)
    ABC = np.cross(AB, AC)

    u = PBC / ABC
    v = PAC / ABC
    w = PAB / ABC

    return np.array([u, v, w])


def ptInTriangle(tri, p):
    """
    Args:
        tri Nx3x2: vertices of triangles
        p 2: query point
    RET:
        N bool, whether the query point is in the triangle (containing edge).
    """
    # f_num = tri.shape[0]
    eps = 1e-9

    AB = tri[:, 1] - tri[:, 0]  # Nx2
    AC = tri[:, 2] - tri[:, 0]
    AP = p - tri[:, 0]

    area = np.cross(AB, AC)  # N
    # sign = np.where(area < 0, -np.ones(f_num), np.ones(f_num))

    s = np.cross(AP, AC) / (area + eps)  # N
    t = np.cross(AB, AP) / (area + eps)

    return s, t, (s >= -1e-4) & (t >= -1e-4) & ((s + t) <= 1 + 1e-4)


def check_uv2vert(obj):
    """check从uv坐标到vert坐标的映射是否为单射,即一个uv坐标对应一个vert坐标

    Args:
        obj['verts']:   7306x3, xyz coordinates of vertices
        obj['uvs']:     32808x2, uv coordinates for uv_ids
        obj['vert_ids']:    10936x3, vert_ids to build faces
        obj['uv_ids']:      10936x3, corresponding uv_ids for vert_ids
    """
    uvs = obj["uvs"][obj["uv_ids"]].reshape(-1, 2)  # 32808x2
    # verts = obj['verts'][obj['vert_ids']].reshape(-1, 3)    # 32808x3
    verts = obj["vert_ids"].reshape(-1)

    assert uvs.shape[0] == verts.shape[0]
    uvs_dict = {}
    for i in range(uvs.shape[0]):
        uv_coord = tuple(uvs[i])
        if uv_coord in uvs_dict:
            if verts[i] not in uvs_dict[uv_coord]:
                uvs_dict[uv_coord].append(verts[i])
        else:
            uvs_dict[uv_coord] = [verts[i]]

    return uvs_dict


def check_seam(obj):
    """
    Args:
        obj['verts']:   7306x3, xyz coordinates of vertices
        obj['uvs']:     32808x2, uv coordinates for uv_ids
        obj['vert_ids']:    10936x3, vert_ids to build faces
        obj['uv_ids']:      10936x3, corresponding uv_ids for vert_ids
    RET:
        vert_uvs:   one-by-one mapping between vert_ids and uvs
    """
    vert_uvs = {}
    for i, face in enumerate(obj["vert_ids"]):
        for j, v_id in enumerate(face):
            uv = obj["uvs"][obj["uv_ids"][i, j]]
            if v_id in vert_uvs.keys() and not (vert_uvs[v_id] == uv).all():
                vert_uvs[v_id].append(uv)
                # return vert_uvs, False
            else:
                vert_uvs[v_id] = [uv]

    return vert_uvs, True


def Mesh2PositionMap(obj, pos_size=256, progress=False):
    """~ 45s
    Args:
        obj['verts']:   7306x3, xyz coordinates of vertices
        obj['uvs']:     32808x2, uv coordinates for uv_ids
        obj['vert_ids']:    10936x3, vert_ids to build faces
        obj['uv_ids']:      10936x3, corresponding uv_ids for vert_ids

        pos_size (int, optional): size of postion map. Defaults to 256.
        progress (bool, optional): if showing progress bar. Defaults is False.
    RET:
        posMap:     256x256x3, postion map of mesh
    """
    posMap = np.zeros((pos_size, pos_size, 3))

    tri = obj["uvs"][obj["uv_ids"]] * (pos_size - 1)  # Nx3x2

    if progress:
        bar = tqdm(range(pos_size))
    for u in range(pos_size):
        for v in range(pos_size):
            p = np.array([u, v])
            s, t, searchRes = ptInTriangle(tri, p)

            # print(np.where(searchRes))
            f_id = np.where(searchRes)[0][0]
            bary = np.array([[(1 - s[f_id] - t[f_id])], [s[f_id]], [t[f_id]]])
            face = obj["vert_ids"][f_id]
            posMap[u, v] = (obj["verts"][face] * bary).sum(axis=0)

        if progress:
            bar.update()

    return posMap




def write_obj(filepath, verts, tris=None, log=True):
    """write obj file

    Args:
        verts:      65536x3, vertices coordinates
        tris:       n_facex3, faces consisting of vertices id
    """
    fw = open(filepath, "w")
    # vertices
    for vert in verts:
        fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

    if not tris is None:
        for tri in tris:
            fw.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    fw.close()
    if log:
        print(f"mesh has been saved in {filepath}.")


def visPositionMap(savepath, posMap):
    """
    Args:
        savepath:   str, path to save
        posMap:     256x256x3, postion map of mesh
    """
    H, W, _ = posMap.shape

    verts = posMap.reshape(-1, 3)
    mmin = verts.min(axis=0)
    mmax = verts.max(axis=0)

    normalized = (verts - mmin) / (mmax - mmin)
    cv2.imwrite(savepath, normalized.reshape(H, W, 3) * 255)


def visDepthMap(savepath, depth_map):
    """
    Args:
        savepath (str): path to save
        depth_map (float): 2048x1334, rendered depth map
    """
    mmin = depth_map[depth_map > 0].min()
    mmax = depth_map[depth_map > 0].max()

    normalized = (depth_map - mmin) / (mmax - mmin)
    normalized[normalized < 0] = 1
    cv2.imwrite(savepath, (1 - normalized) * 255)


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and p.grad is not None and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")

    plt.savefig(fname="./test.png")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

